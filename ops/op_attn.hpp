#pragma once

#include <stdint.h>
#include "tensor.hpp"
#include "utils.hpp"

void rope_embed(tensor& x, tensor inv_freq, int position_id) {
  // assume x : [B, H, L, S]
  auto B = x.size(0);
  auto H = x.size(1);
  auto L = x.size(2);
  auto S = x.size(3);
  // std::cout << "       x: " << x.repr(false) << "\n";
  // std::cout << "inv_freq: " << inv_freq.repr(false) << "\n";
  auto half_ro_ndims = inv_freq.size(0);
  auto* ifreq = inv_freq.data<float>();
  parallel_nt(0, B * H, 0, [&](int64_t bh0, int64_t bh1) {
    for (auto bh = bh0; bh < bh1; bh++) {
      auto b = bh / H;
      auto h = bh % H;
      for (int k = 0; k < L; k++) {
        float* px = &x.at<float>({b, h, k, 0});
        int i0 = 0;
        for (i0 = 0; i0 < half_ro_ndims; i0++) {
          auto i1 = i0 + half_ro_ndims;
          float xita = ifreq[i0] * (k + position_id);
          float vcos = std::cos(xita);
          float vsin = std::sin(xita);
          auto x0 = px[i0];
          auto x1 = px[i1];
          auto y0 = vcos * x0 - vsin * x1;
          auto y1 = vsin * x0 + vcos * x1;
          px[i0] = y0;
          px[i1] = y1;
        }
      }
    }
  });
}

// attention with RoPE and kv-cache
void attention_rope(tensor q,          // [B, qL, H*S]
                    tensor k,          // [B, qL, H*S]
                    tensor v,          // [B, qL, H*S]
                    tensor inv_freq,   // [rotary_dims/2] for RoPE
                    tensor kv_cache,   // [2, B, H, max_length, S]
                    tensor kvc_slots,  // [qL]
                    int position_id,
                    int layer_id) {
  // validate dtype & rank
  ASSERT(q.is<float>(3));
  ASSERT(k.is<float>(3));
  ASSERT(v.is<float>(3));
  ASSERT(kv_cache.is<float>(5));
  ASSERT(kvc_slots.is<long>(1));

  auto B = q.size(0);
  auto qL = q.size(1);
  auto H = kv_cache.size(2);
  auto max_kv_length = kv_cache.size(3);
  auto S = kv_cache.size(-1);

  // validate shape
  ASSERT(q.is({B, qL, H * S}));
  ASSERT(k.is({B, qL, H * S}));
  ASSERT(v.is({B, qL, H * S}));
  // ASSERT(kv_cache.is({2ll, B, H, max_length, S}));
  ASSERT(kvc_slots.is({qL}));

  q = q.reshape({B, qL, H, S}).permute({0, 2, 1, 3});  // B,H,qL,S
  k = k.reshape({B, qL, H, S}).permute({0, 2, 1, 3});
  v = v.reshape({B, qL, H, S}).permute({0, 2, 1, 3});

  // inplace positional embedding
  rope_embed(q, inv_freq, position_id);
  rope_embed(k, inv_freq, position_id);

  // put k/v into cache
  auto* slots = kvc_slots.data<long>();
  parallel_nt(0, B * H * qL, 0, [&](int64_t bhl0, int64_t bhl1) {
    for (auto bhl = bhl0; bhl < bhl1; bhl++) {
      auto pk = bhl % qL;
      auto bh = (bhl / qL);
      auto h = bh % H;
      auto b = bh / H;
      memcpy(&kv_cache.at<float>({2 * layer_id + 0, b, h, slots[pk], 0}),
             &k.at<float>({b, h, pk, 0}), sizeof(float) * S);
      memcpy(&kv_cache.at<float>({2 * layer_id + 1, b, h, slots[pk], 0}),
             &v.at<float>({b, h, pk, 0}), sizeof(float) * S);
    }
  });

  auto kcache = kv_cache.slice(0, 2 * layer_id + 0, 2 * layer_id + 0);
  auto vcache = kv_cache.slice(0, 2 * layer_id + 1, 2 * layer_id + 1);

  auto kvLen = position_id + qL;
  if (kvLen > max_kv_length) {
    kvLen = max_kv_length;
  }

  // main attention logic
  auto d_scale = 1.0f / sqrt(S);
  tensor attn_w;
  attn_w.reset(static_cast<float*>(nullptr), {B, H, qL, kvLen});
  parallel_nt(0, B * H * kvLen, 0, [&](int64_t bhl0, int64_t bhl1) {
    for (auto bhl = bhl0; bhl < bhl1; bhl++) {
      auto pk = bhl % kvLen;
      auto bh = (bhl / kvLen);
      auto h = bh % H;
      auto b = bh / H;
      for (int64_t pq = 0; pq < qL; pq++) {
        float sum = 0;
        sum = dot_product(&q.at<float>({b, h, pq, 0}),
                          &kcache.at<float>({b, h, pk, 0}), S);
        attn_w.at<float>({b, h, pq, pk}) = sum * d_scale;
      }
    }
  });

  // softmax
  parallel_nt(0, B * H * qL, 0, [&](int64_t bhl0, int64_t bhl1) {
    for (auto bhl = bhl0; bhl < bhl1; bhl++) {
      auto pq = bhl % qL;
      auto bh = (bhl / qL);
      auto h = bh % H;
      auto b = bh / H;

      // clear invalid kv attention weights
      auto* pw = &attn_w.at<float>({b, h, pq, 0});
      for (int64_t pk = pq + 1; pk < qL; pk++) {
        pw[slots[pk]] = std::numeric_limits<float>::lowest();
      }
      _softmax(pw, kvLen);
    }
  });

  tensor m_temp;
  auto nthr = omp_get_max_threads();
  m_temp.reset(static_cast<float*>(nullptr), {nthr, B, qL, H, S});

  int64_t work_amount = B * H * kvLen;
  parallel_nt(0, nthr, 0, [&](int64_t tid0, int64_t tid1) {
    auto ithr = omp_get_thread_num();
    memset(&m_temp.at<float>({ithr, 0, 0, 0, 0}), 0,
           m_temp.stride(0) * sizeof(float));

    // split works amount nthr
    int64_t i0, i1;
    splitter(work_amount, nthr, ithr, i0, i1);
    for (int64_t i = i0; i < i1; i++) {
      auto pv = i % kvLen;
      auto bh = i / kvLen;
      auto h = bh % H;
      auto b = bh / H;
      auto* v = &vcache.at<float>({b, h, pv, 0});
      for (int64_t pq = 0; pq < qL; pq++) {
        auto* out = &m_temp.at<float>({ithr, b, pq, h, 0});
        auto weight = attn_w.at<float>({b, h, pq, pv});
        accumulate_weighted_v(out, weight, v, S);
      }
    }
  });

  // inplace output to q (override q)
  parallel_nt(0, B * H * qL, 0, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; i++) {
      auto pq = i % qL;
      auto bh = i / qL;
      auto h = bh % H;
      auto b = bh / H;
      auto* temp = &m_temp.at<float>({0, b, pq, h, 0});
      size_t temp_stride = m_temp.stride(0);
      auto* dst = &q.at<float>({b, h, pq, 0});
      reduce_v(dst, temp, nthr, S, temp_stride);
    }
  });
}
