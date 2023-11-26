#pragma once

#include <stdint.h>
#include "tensor.hpp"
#include "utils.hpp"
/*
        def rope_embedd(x):
            half_rotary_dim = self.rotary_dims//2
            for k in range(q_len):
                cur_position_id = position_id + k

                # better for python
                xita = inv_freq * cur_position_id
                vcos = torch.cos(xita)
                vsin = torch.sin(xita)
                x0 = x[:, :, k, :half_rotary_dim]
                x1 = x[:, :, k, half_rotary_dim:]
                y0 = vcos * x0 - vsin * x1
                y1 = vsin * x0 + vcos * x1
                x[:, :, k, :half_rotary_dim] = y0
                x[:, :, k, half_rotary_dim:] = y1

                ## better for C++
                #for i0 in range(half_rotary_dim):
                #    i1 = i0 + half_rotary_dim
                #   xita = (inv_freq[i0] * cur_position_id)
                #    vcos = math.cos(xita)
                #    vsin = math.sin(xita)
                #    y0 = vcos * x[:, :, k, i0] - vsin * x[:, :, k, i1]
                #    y1 = vsin * x[:, :, k, i0] + vcos * x[:, :, k, i1]
                #    x[:, :, k, i0] = y0
                #    x[:, :, k, i1] = y1
*/
void rope_embed(tensor& x, tensor inv_freq, int position_id) {
  // assume x : [B, H, L, S]
  auto B = x.size(0);
  auto H = x.size(1);
  auto L = x.size(2);
  auto S = x.size(3);
  auto half_ro_ndims = inv_freq.size(0);
  auto* ifreq = inv_freq.data<float>();
  parallel_nt(0, B * H, 0, [&](int64_t bh0, int64_t bh1) {
    for (auto bh = bh0; bh < bh1; bh++) {
      auto b = bh / H;
      auto h = bh % H;
      for (int k = 0; k < L; k++) {
        auto* px = &x.at<float>({b, h, k, 0});
        int i0 = 0;
        for (i0 = 0; i0 < half_ro_ndims; i0++) {
          auto i1 = i0 + half_ro_ndims;
          auto xita = ifreq[i0] * (k + position_id);
          auto vcos = std::cos(xita);
          auto vsin = std::sin(xita);
          auto& x0 = px[i0];
          auto& x1 = px[i1];
          auto y0 = vcos * x0 - vsin * x1;
          auto y1 = vsin * x0 + vcos * x1;
          x0 = y0;
          x1 = y1;
        }
      }
    }
  });
}
#if 0
// attention with RoPE and kv-cache
void attention_rope(tensor output,     // [B, qL, H*S]
                    tensor q,          // [B, qL, H*S]
                    tensor k,          // [B, kvL, H*S]
                    tensor v,          // [B, kvL, H*S]
                    tensor inv_freq,   // [rotary_dims/2] for RoPE
                    tensor kv_cache,   // [2, B, H, max_length, S]
                    tensor kvc_slots,  // [kvL]
                    int position_id) {
  // validate dtype & rank
  ASSERT(output.is<float>(3));
  ASSERT(q.is<float>(3));
  ASSERT(k.is<float>(3));
  ASSERT(v.is<float>(3));
  ASSERT(kv_cache.is<float>(5));
  ASSERT(kvc_slots.is<int32_t>(1));

  auto B = q.size(0);
  auto qL = q.size(1);
  auto H = kv_cache.size(2);
  auto max_length = kv_cache.size(3);
  auto S = kv_cache.size(-1);
  auto kvL = k.size(1);

  // validate shape
  ASSERT(q.is({B, qL, H * S}));
  ASSERT(k.is({B, kvL, H * S}));
  ASSERT(v.is({B, kvL, H * S}));
  ASSERT(kv_cache.is({2, B, H, max_length, S}));
  ASSERT(kvc_slots.is({kvL}));
  ASSERT(output.is({B, qL, H * S}));

  // positional embedding

}
#endif
