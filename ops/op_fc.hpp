#pragma once

#include <stdint.h>
#include "tensor.hpp"
#include "utils.hpp"

//======================================================================================
// fullyconnect

// deq(q)=(d*q + m)
struct q4_1_block {
  // whole 4-bit 32x32 block distributed as following
  //    8x(4x32) each 2 adjacent (4x32) is combined into low/high part of a 8bit
  //    4x32
  int8_t w[4][32 * 4];
  float16 wd[32];
  float16 wm[32];

  void set(int k, int n, int8_t v) {
    // assert(v >= -8 && v < 7)
    auto& value = w[k >> 3][(n * 4) + (k & 3)];
    bool is_high_4bit = ((k / 4) & 1);
    if (is_high_4bit) {
      value = (value & 0x0F) | (v << 4);
    } else {
      value = (value & 0xF0) | (v & 0x0F);
    }
  }
  float get(int k, int n) {
    auto& value = w[k >> 3][(n * 4) + (k & 3)];
    bool is_high_4bit = ((k / 4) & 1);
    int8_t q;
    if (is_high_4bit) {
      q = (value >> 4) & 0xF;
    } else {
      q = (value & 0xF);
    }
    auto d = to_fp32(wd[n]);
    auto m = to_fp32(wm[n]);
    return d * q + m;
  }
};

static inline void get_min_max(float* x, int len, float& vmin, float& vmax) {
  int i = 1;
  vmin = vmax = x[0];
  for (; i < len; i++) {
    auto a = x[i];
    if (vmax < a)
      vmax = a;
    if (vmin > a)
      vmin = a;
  }
}

tensor offline_FC_dequant_Q4A(tensor weiq) {
  ASSERT(weiq.is<int8_t>(3));
  int64_t Ngroups = weiq.size(0);
  int64_t Kgroups = weiq.size(1);
  int64_t group_k = 32;
  int64_t group_n = 32;
  ASSERT(weiq.size(2) == sizeof(q4_1_block));

  tensor output;
  output.reset<float>(nullptr, {Ngroups * group_n, Kgroups * group_k});

  parallel_nt(0, Ngroups, 0, [&](int64_t nb0, int64_t nb1) {
    for (auto nb = nb0; nb < nb1; nb++) {
      auto n0 = nb * group_n;
      q4_1_block* wq4 = &weiq.at<q4_1_block>({nb, 0, 0});
      for (int64_t kb = 0, k0 = 0; kb < Kgroups; kb++, k0 += group_k, wq4++) {
        for (int64_t ni = 0; ni < group_n; ni++) {
          for (int64_t ki = 0; ki < group_k; ki++) {
            output.at<float>({n0 + ni, k0 + ki}) = wq4->get(ki, ni);
          }
        }
      }
    }
  });
  return output;
}

tensor offline_FC_quant_Q4A(tensor wei) {
  ASSERT(wei.is<float>(2));
  // raw weight input is NxK (transpose_b is true)
  // strides is decreasing, so inner-most dimension is at higher ranks
  int64_t N = wei.size(0);
  int64_t K = wei.size(1);
  int64_t group_k = 32;
  int64_t group_n = 32;
  int64_t Kgroups = (K + group_k - 1) / group_k;
  int64_t Ngroups = (N + group_n - 1) / group_n;

  tensor wei_quantized;
  wei_quantized.reset<int8_t>(
      nullptr, {Ngroups, Kgroups, static_cast<int64_t>(sizeof(q4_1_block))});

  // each 32x32 sub-block is further interleaved every 4-rows into (32/4)x(32*4)
  // and each column of 32x32 sub-block share a quantization scales
  parallel_nt(0, Ngroups, 0, [&](int64_t nb0, int64_t nb1) {
    for (auto nb = nb0; nb < nb1; nb++) {
      auto n0 = nb * group_n;
      q4_1_block* wq4 = &wei_quantized.at<q4_1_block>({nb, 0, 0});
      for (int64_t kb = 0, k0 = 0; kb < Kgroups; kb++, k0 += group_k, wq4++) {
        // w_q composed of
        for (int64_t ni = 0; ni < group_n; ni++) {
          // asymmetric quantization
          auto src_n = n0 + ni;
          if (src_n >= N) {
            wq4->wd[ni] = 0;
            wq4->wm[ni] = 0;
            continue;
          }

          float vmin, vmax;
          get_min_max(&wei.at<float>({src_n, k0}), group_k, vmin, vmax);

          //  to use deq(q)=(d*q + m) to map (vmin,vmax) to 0-15
          //     d = (vmax-vmin)/15
          //     m = vmin
          const float level_max = 15.0f;
          float d = (vmax - vmin) / level_max;
          float m = vmin;
          float id = (d != 0) ? (1.0f / d) : 0;

          wq4->wd[ni] = to_fp16(d);
          wq4->wm[ni] = to_fp16(m);

          for (int ki = 0; ki < group_k; ki++) {
            auto src_k = k0 + ki;
            int8_t w_quantized = 0;
            if (src_k < K) {
              auto w_round =
                  std::roundf((wei.at<float>({src_n, src_k}) - m) * id);
              w_quantized = std::min(level_max, std::max(w_round, 0.0f));
            }
            wq4->set(ki, ni, w_quantized);
          }
        }
      }
    }
  });
  return wei_quantized;
}

void FC_dynamic_quantize_x(tensor& input,
                           tensor& x_quantized,
                           tensor& x_scales,
                           tensor* px_group_sum,
                           int64_t Kgroups,
                           int64_t group_k,
                           float scale) {
  auto B = input.size(0);
  auto M = input.size(1);
  auto K = input.size(2);
  // dynamically quantize whole inputs
  // x_quantized.resize({B, M, Kgroups * group_k});
  // x_scales.resize({B, M, Kgroups});
  // x_group_sum.resize({B, M, Kgroups});

  // kernel is light-weight to parallel, unless we have multiple rows
  parallel_nt(0, B * M, 0, [&](int64_t bm0, int64_t bm1) {
    for (auto bm = bm0; bm < bm1; bm++) {
      auto b = bm / M;
      auto m = bm % M;
      // a single row quantized in K groups
      float* q8_xd = &x_scales.at<float>({b, m, 0});
      float* x_gsum =
          px_group_sum ? &(px_group_sum->at<float>({b, m, 0})) : nullptr;
      int8_t* q8_xq = &x_quantized.at<int8_t>({b, m, 0});
      float* raw_x = &input.at<float>({b, m, 0});
      for (int64_t kb = 0, left_k = K; kb < Kgroups;
           kb++, raw_x += group_k, q8_xq += group_k, left_k -= group_k) {
        auto actual_len = std::min(group_k, left_k);
        auto amax = get_amax(raw_x, actual_len);
        // x = (d * quantized)
        // quantized = round(x / d) = round(x * id)
        const float d = amax / 127;
        const float id = (d != 0) ? (1.0f / d) : 0;

        q8_xd[kb] = d * scale;
        quant_row_q8_0(raw_x, q8_xq, actual_len, id);

        // fill zero to the padding part
        if (actual_len < group_k) {
          memset(q8_xq + actual_len, 0, group_k - actual_len);
        }
        if (x_gsum) {
          float group_sum = 0.0f;
          for (int ki = 0; ki < actual_len; ki++) {
            group_sum += raw_x[ki];
          }
          x_gsum[kb] = group_sum;
        }
      }
    }
  });
}

/*****************************************************************************
target      : sum(Xi * Wi)
approximate : Xi ~ (Sx*Qxi)         // Qxi is 8bits signed
              Wi ~ (Sw*Qwi + m)     // Qwi is 2bits unsigned

result      : sum[Sx*Qxi * (Sw*Qwi + m)] = (Sx*Sw) * sum(Qxi*Qwi) + m *
sum(Sx*Qxi)

    sum(Qxi*Qwi) is calculated using AVX_VNNI
    sum(Sx*Qxi) is dynamically pre-calculated
*******************************************************************************/
tensor fc_Q4A(tensor input, tensor wei_quantized, int N) {
  ASSERT(input.is<float>(3));
  ASSERT(wei_quantized.is<int8_t>(3));
  auto Ngroups = wei_quantized.size(0);
  auto Kgroups = wei_quantized.size(1);
  int64_t group_k = 32;
  int64_t group_n = 32;
  auto B = input.size(0);
  auto M = input.size(1);
  auto K = input.size(2);
  VNNI_Sequence vnni_raw;

  tensor output;
  output.reset<float>(nullptr, {B, M, N});

  auto y_stride = output.stride(1);

  // dynamically quantize whole inputs
  tensor x_quantized;
  tensor x_scales;
  tensor x_group_sum;

  x_quantized.reset<int8_t>(nullptr, {B, M, Kgroups * group_k});
  x_scales.reset<float>(nullptr, {B, M, Kgroups});
  x_group_sum.reset<float>(nullptr, {B, M, Kgroups});

  FC_dynamic_quantize_x(input, x_quantized, x_scales, &x_group_sum, Kgroups,
                        group_k, 1.0f);

  parallel_nt(0, Ngroups, 0, [&](int64_t nb0, int64_t nb1) {
    for (auto nb = nb0; nb < nb1; nb++) {
      auto n0 = nb * group_n;
      float* py = &output.at<float>({0, 0, n0});
      // B & M dimensions are collapsed as 1 dimension
      for (int64_t b = 0; b < B; b++) {
        for (int64_t m = 0; m < M; m++, py += y_stride) {
          const float* q8_xd = &x_scales.at<float>({b, m, 0});
          const float* xg_sum = &x_group_sum.at<float>({b, m, 0});
          const int8_t* q8_xq = &x_quantized.at<int8_t>({b, m, 0});

          __m256 acc0 = _mm256_setzero_ps();
          __m256 acc1 = _mm256_setzero_ps();
          __m256 acc2 = _mm256_setzero_ps();
          __m256 acc3 = _mm256_setzero_ps();

          const q4_1_block* wq4 = &wei_quantized.at<q4_1_block>({nb, 0, 0});
          for (int kb = 0, k0 = 0; kb < Kgroups;
               kb++, k0 += group_k, q8_xd++, wq4++, xg_sum++) {
            // K group is smallest quantization unit which shares single scale
            auto acci0 = _mm256_setzero_si256();
            auto acci1 = _mm256_setzero_si256();
            auto acci2 = _mm256_setzero_si256();
            auto acci3 = _mm256_setzero_si256();
            const __m256i low4_mask = _mm256_set1_epi32(0x0F0F0F0F);
            auto* q4_weight = wq4->w[0];
            for (int ki = 0; ki < group_k;
                 ki += 8, q4_weight += 32 * 4, q8_xq += 8) {
              // low 4bit 4x32 blocks
              __m256i x0 =
                  _mm256_set1_epi32(*reinterpret_cast<const int32_t*>(q8_xq));
              __m256i y0 = _mm256_loadu_si256((const __m256i*)(q4_weight));
              __m256i y1 = _mm256_loadu_si256((const __m256i*)(q4_weight + 32));
              __m256i y2 =
                  _mm256_loadu_si256((const __m256i*)(q4_weight + 32 * 2));
              __m256i y3 =
                  _mm256_loadu_si256((const __m256i*)(q4_weight + 32 * 3));

              acci0 = vnni_raw(acci0, x0, _mm256_and_si256(y0, low4_mask));
              acci1 = vnni_raw(acci1, x0, _mm256_and_si256(y1, low4_mask));
              acci2 = vnni_raw(acci2, x0, _mm256_and_si256(y2, low4_mask));
              acci3 = vnni_raw(acci3, x0, _mm256_and_si256(y3, low4_mask));

              // high 4bit
              __m256i x1 = _mm256_set1_epi32(
                  *reinterpret_cast<const int32_t*>(q8_xq + 4));
              acci0 = vnni_raw(
                  acci0, x1,
                  _mm256_and_si256(_mm256_srli_epi16(y0, 4), low4_mask));
              acci1 = vnni_raw(
                  acci1, x1,
                  _mm256_and_si256(_mm256_srli_epi16(y1, 4), low4_mask));
              acci2 = vnni_raw(
                  acci2, x1,
                  _mm256_and_si256(_mm256_srli_epi16(y2, 4), low4_mask));
              acci3 = vnni_raw(
                  acci3, x1,
                  _mm256_and_si256(_mm256_srli_epi16(y3, 4), low4_mask));
            }
            // load de-quantize coeff and combine with input's dequantize coeff
            // const u_int16_t *f16_scale = reinterpret_cast<const u_int16_t
            // *>(&wei_scales({kb, n0}));
            auto dx = _mm256_broadcast_ss(q8_xd);

            auto d0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)wq4->wd));
            auto d1 = _mm256_cvtph_ps(
                _mm_loadu_si128((const __m128i*)(&wq4->wd[8 * 1])));
            auto d2 = _mm256_cvtph_ps(
                _mm_loadu_si128((const __m128i*)(&wq4->wd[8 * 2])));
            auto d3 = _mm256_cvtph_ps(
                _mm_loadu_si128((const __m128i*)(&wq4->wd[8 * 3])));

            d0 = _mm256_mul_ps(d0, dx);
            d1 = _mm256_mul_ps(d1, dx);
            d2 = _mm256_mul_ps(d2, dx);
            d3 = _mm256_mul_ps(d3, dx);

            // dequantize
            acc0 = _mm256_fmadd_ps(d0, _mm256_cvtepi32_ps(acci0), acc0);
            acc1 = _mm256_fmadd_ps(d1, _mm256_cvtepi32_ps(acci1), acc1);
            acc2 = _mm256_fmadd_ps(d2, _mm256_cvtepi32_ps(acci2), acc2);
            acc3 = _mm256_fmadd_ps(d3, _mm256_cvtepi32_ps(acci3), acc3);

            // compensation term caused by zero-points
            auto m0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)wq4->wm));
            auto m1 = _mm256_cvtph_ps(
                _mm_loadu_si128((const __m128i*)(&wq4->wm[8 * 1])));
            auto m2 = _mm256_cvtph_ps(
                _mm_loadu_si128((const __m128i*)(&wq4->wm[8 * 2])));
            auto m3 = _mm256_cvtph_ps(
                _mm_loadu_si128((const __m128i*)(&wq4->wm[8 * 3])));

            auto gsum = _mm256_broadcast_ss(xg_sum);
            acc0 = _mm256_fmadd_ps(m0, gsum, acc0);
            acc1 = _mm256_fmadd_ps(m1, gsum, acc1);
            acc2 = _mm256_fmadd_ps(m2, gsum, acc2);
            acc3 = _mm256_fmadd_ps(m3, gsum, acc3);
          }

          // output 32 results
          _mm256_storeu_ps(py + 8 * 0, acc0);
          _mm256_storeu_ps(py + 8 * 1, acc1);
          _mm256_storeu_ps(py + 8 * 2, acc2);
          _mm256_storeu_ps(py + 8 * 3, acc3);
        }
      }
    }
  });

  return output;
}
