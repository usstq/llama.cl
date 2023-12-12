#pragma once

#include <stdint.h>

#include "profiler.hpp"
#include "tensor.hpp"
#include "utils.hpp"

struct CosSin_cache {
    tensor cache;
    int64_t cur_len;
    CosSin_cache() {
        cur_len = 0;
    }
    void update(int len, tensor inv_freq) {
        int64_t half_ro_ndims = inv_freq.size(0);
        if (cur_len < len) {
            if (cur_len == 0)
                cur_len = 1;
            while (cur_len < len) {
                cur_len *= 2;
            }
            cache.reset<float>(nullptr, {cur_len, 2, half_ro_ndims});
            auto* ifreq = inv_freq.data<float>();
            for (int64_t pos = 0; pos < cur_len; pos++) {
                float* cos_cache = &cache.at<float>({pos, 0, 0});
                float* sin_cache = &cache.at<float>({pos, 1, 0});
                for (int64_t i0 = 0; i0 < half_ro_ndims; i0++) {
                    float xita = ifreq[i0] * pos;
                    cos_cache[i0] = std::cos(xita);
                    sin_cache[i0] = std::sin(xita);
                }
            }
        }
    }
    float* get_cos(int pos) {
        return &cache.at<float>({pos, 0, 0});
    }
    float* get_sin(int pos) {
        return &cache.at<float>({pos, 1, 0});
    }
};

void rope_embed(tensor& x, tensor inv_freq, int position_id) {
    static CosSin_cache cs_cache;

    // assume x : [B, H, L, S]
    auto B = x.size(0);
    auto H = x.size(1);
    auto L = x.size(2);
    auto S = x.size(3);
    // std::cout << "       x: " << x.repr(false) << "\n";
    // std::cout << "inv_freq: " << inv_freq.repr(false) << "\n";
    auto half_ro_ndims = inv_freq.size(0);
    auto* ifreq = inv_freq.data<float>();
    cs_cache.update(position_id + L, inv_freq);

    parallel_nt(0, B * H, 0, [&](int64_t bh0, int64_t bh1) {
        for (auto bh = bh0; bh < bh1; bh++) {
            auto b = bh / H;
            auto h = bh % H;
            for (int k = 0; k < L; k++) {
                float* px = &x.at<float>({b, h, k, 0});
                float* pcos = cs_cache.get_cos(k + position_id);
                float* psin = cs_cache.get_sin(k + position_id);
                int i0 = 0;
#if __AVX2__
                for (i0 = 0; i0 + 8 <= half_ro_ndims; i0 += 8) {
                    auto i1 = i0 + half_ro_ndims;
                    auto vcos = _mm256_loadu_ps(pcos + i0);
                    auto vsin = _mm256_loadu_ps(psin + i0);
                    auto x0 = _mm256_loadu_ps(px + i0);
                    auto x1 = _mm256_loadu_ps(px + i1);
                    auto y0 = _mm256_sub_ps(_mm256_mul_ps(vcos, x0), _mm256_mul_ps(vsin, x1));
                    auto y1 = _mm256_add_ps(_mm256_mul_ps(vsin, x0), _mm256_mul_ps(vcos, x1));
                    _mm256_storeu_ps(px + i0, y0);
                    _mm256_storeu_ps(px + i1, y1);
                }
#endif
                for (; i0 < half_ro_ndims; i0++) {
                    auto i1 = i0 + half_ro_ndims;
                    float vcos = pcos[i0];
                    float vsin = psin[i0];
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

void rope_embed_to(tensor& x, tensor& y, int32_t* y_slots, tensor inv_freq, int position_id) {
    static CosSin_cache cs_cache;

    // assume x : [B, H, L, S]
    auto B = x.size(0);
    auto H = x.size(1);
    auto L = x.size(2);
    auto S = x.size(3);
    // std::cout << "       x: " << x.repr(false) << "\n";
    // std::cout << "inv_freq: " << inv_freq.repr(false) << "\n";
    auto half_ro_ndims = inv_freq.size(0);
    auto* ifreq = inv_freq.data<float>();
    cs_cache.update(position_id + L, inv_freq);

    parallel_nt(0, B * H, 0, [&](int64_t bh0, int64_t bh1) {
        for (auto bh = bh0; bh < bh1; bh++) {
            auto b = bh / H;
            auto h = bh % H;
            for (int k = 0; k < L; k++) {
                float* px = &x.at<float>({b, h, k, 0});
                float* py = &y.at<float>({b, h, y_slots[k], 0});
                float* pcos = cs_cache.get_cos(k + position_id);
                float* psin = cs_cache.get_sin(k + position_id);
                int i0 = 0;
#if __AVX2__
                for (i0 = 0; i0 + 8 <= half_ro_ndims; i0 += 8) {
                    auto i1 = i0 + half_ro_ndims;
                    auto vcos = _mm256_loadu_ps(pcos + i0);
                    auto vsin = _mm256_loadu_ps(psin + i0);
                    auto x0 = _mm256_loadu_ps(px + i0);
                    auto x1 = _mm256_loadu_ps(px + i1);
                    auto y0 = _mm256_sub_ps(_mm256_mul_ps(vcos, x0), _mm256_mul_ps(vsin, x1));
                    auto y1 = _mm256_add_ps(_mm256_mul_ps(vsin, x0), _mm256_mul_ps(vcos, x1));
                    _mm256_storeu_ps(py + i0, y0);
                    _mm256_storeu_ps(py + i1, y1);
                }
#endif
                for (; i0 < half_ro_ndims; i0++) {
                    auto i1 = i0 + half_ro_ndims;
                    float vcos = pcos[i0];
                    float vsin = psin[i0];
                    auto x0 = px[i0];
                    auto x1 = px[i1];
                    auto y0 = vcos * x0 - vsin * x1;
                    auto y1 = vsin * x0 + vcos * x1;
                    py[i0] = y0;
                    py[i1] = y1;
                }
            }
        }
    });
}

tensor mm_qk(tensor q,      // [B, qL, H*S]
             tensor kcache  // [B, H, kvLen, S]
) {
    auto B = q.size(0);
    auto qL = q.size(1);
    ASSERT(kcache.size(-4) == B);
    auto H = kcache.size(-3);
    auto kvLen = kcache.size(-2);
    auto S = kcache.size(-1);

    q = q.reshape({B, qL, H, S}).permute({0, 2, 1, 3});  // B,H,qL,S

    auto d_scale = 1.0f / std::sqrt(S);
    tensor attn_w;
    attn_w.reset(static_cast<float*>(nullptr), {B, H, qL, kvLen});

    auto qk_kernel = [&](int64_t b, int64_t h, int64_t pq0, int64_t pq1, int64_t pk0, int64_t pk1) {
        for (int64_t pq = pq0; pq < pq1; pq++) {
            for (int64_t pk = pk0; pk < pk1; pk++) {
                float sum = 0;
                sum = dot_product(&q.at<float>({b, h, pq, 0}), &kcache.at<float>({b, h, pk, 0}), S);
                attn_w.at<float>({b, h, pq, pk}) = sum * d_scale;
            }
        }
    };
    parallel_nt(0, B * H, 0, [&](int64_t bh0, int64_t bh1) {
        constexpr int block = 128;
        for (auto bh = bh0; bh < bh1; bh++) {
            auto h = bh % H;
            auto b = bh / H;
            for (int64_t pq0 = 0; pq0 < qL; pq0 += block) {
                auto pq1 = pq0 + block;
                if (pq1 > qL)
                    pq1 = qL;
                for (int64_t pk0 = 0; pk0 < kvLen; pk0 += block) {
                    auto pk1 = pk0 + block;
                    if (pk1 > kvLen)
                        pk1 = kvLen;
                    qk_kernel(b, h, pq0, pq1, pk0, pk1);
                }
            }
        }
    });
    return attn_w;
}

void mm_qk42(tensor& q,       // [B, H, qL, S]
             tensor& kcache,  // [B, H, max-kv-Len, S]
             tensor& attn_w   // [B, H, qL, kvLen]
) {
    auto B = q.size(0);
    auto qL = q.size(2);
    auto H = kcache.size(-3);
    auto kvLen = attn_w.size(-1);
    auto S = kcache.size(-1);
    ASSERT(kcache.size(-4) == B);
    ASSERT((S % 8) == 0);

    float d_scale = 1.0f / std::sqrt(S);

    auto qk_kernel_4x2 = [&](int64_t b, int64_t h, int64_t pq0, int64_t pq1, int64_t pk0, int64_t pk1) {
        auto* q_ptr0 = &q.at<float>({b, h, pq0, 0});
        auto q_stride = q.stride(2);
        auto q_stride3 = q.stride(2) * 3;
        auto* k_ptr_base = &kcache.at<float>({b, h, pk0, 0});
        auto w_stride = attn_w.stride(2);
        auto* pw_base = &attn_w.at<int32_t>({b, h, pq0, 0});
        auto k_stride = kcache.stride(2);
        for (int64_t pq = pq0; pq < pq1; pq += 4, q_ptr0 += 4 * q_stride, pw_base += 4 * w_stride) {
            auto* k_ptr = k_ptr_base;
            for (int64_t pk = pk0; pk < pk1; pk += 2, k_ptr += 2 * k_stride) {
                auto* q_ptr = q_ptr0;
                auto* k_ptr0 = k_ptr;
                auto* k_ptr1 = k_ptr + k_stride;

                auto vsum0 = _mm256_setzero_ps();
                auto vsum1 = _mm256_setzero_ps();
                auto vsum2 = _mm256_setzero_ps();
                auto vsum3 = _mm256_setzero_ps();
                auto vsum4 = _mm256_setzero_ps();
                auto vsum5 = _mm256_setzero_ps();
                auto vsum6 = _mm256_setzero_ps();
                auto vsum7 = _mm256_setzero_ps();

                for (int64_t i = 0; i + 8 <= S; i += 8, q_ptr += 8) {
                    auto va0 = mm256_uni_loadu_ps(q_ptr);
                    auto va1 = mm256_uni_loadu_ps(q_ptr + q_stride);
                    auto va2 = mm256_uni_loadu_ps(q_ptr + q_stride * 2);
                    auto va3 = mm256_uni_loadu_ps(q_ptr + q_stride3);

                    auto vb0 = mm256_uni_loadu_ps(k_ptr0 + i);
                    vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
                    vsum1 = _mm256_fmadd_ps(va1, vb0, vsum1);
                    vsum2 = _mm256_fmadd_ps(va2, vb0, vsum2);
                    vsum3 = _mm256_fmadd_ps(va3, vb0, vsum3);

                    auto vb1 = mm256_uni_loadu_ps(k_ptr1 + i);
                    vsum4 = _mm256_fmadd_ps(va0, vb1, vsum4);
                    vsum5 = _mm256_fmadd_ps(va1, vb1, vsum5);
                    vsum6 = _mm256_fmadd_ps(va2, vb1, vsum6);
                    vsum7 = _mm256_fmadd_ps(va3, vb1, vsum7);
                }

                transpose8_ps(vsum0, vsum1, vsum2, vsum3, vsum4, vsum5, vsum6, vsum7);
                vsum0 = _mm256_add_ps(vsum0, vsum1);
                vsum2 = _mm256_add_ps(vsum2, vsum3);
                vsum4 = _mm256_add_ps(vsum4, vsum5);
                vsum6 = _mm256_add_ps(vsum6, vsum7);
                vsum0 = _mm256_add_ps(vsum0, vsum2);
                vsum4 = _mm256_add_ps(vsum4, vsum6);
                vsum0 = _mm256_add_ps(vsum0, vsum4);

                vsum0 = _mm256_mul_ps(vsum0, _mm256_broadcast_ss(&d_scale));
                __m128 hi = _mm256_extractf128_ps(vsum0, 1);
                __m128 lo = _mm256_castps256_ps128(vsum0);

                auto* pw = pw_base + pk;

                pw[0] = _mm_extract_ps(lo, 0);
                pw[w_stride] = _mm_extract_ps(lo, 1);
                pw[w_stride * 2] = _mm_extract_ps(lo, 2);
                pw[w_stride * 3] = _mm_extract_ps(lo, 3);

                pw[1] = _mm_extract_ps(hi, 0);
                pw[1 + w_stride] = _mm_extract_ps(hi, 1);
                pw[1 + w_stride * 2] = _mm_extract_ps(hi, 2);
                pw[1 + w_stride * 3] = _mm_extract_ps(hi, 3);
            }
        }
    };
    parallel_nt(0, B * H, 0, [&](int64_t bh0, int64_t bh1) {
        const int block_q = 64;
        const int block_k = 64;
        for (auto bh = bh0; bh < bh1; bh++) {
            auto h = bh % H;
            auto b = bh / H;
            for (int64_t pq0 = 0; pq0 < qL; pq0 += block_q) {
                auto pq1 = pq0 + block_q;
                if (pq1 > qL)
                    pq1 = qL;
                for (int64_t pk0 = 0; pk0 < kvLen; pk0 += block_k) {
                    auto pk1 = pk0 + block_k;
                    if (pk1 > kvLen)
                        pk1 = kvLen;
                    qk_kernel_4x2(b, h, pq0, pq1, pk0, pk1);
                }
            }
        }
    });
}

tensor mm_qk81(tensor& q,      // [B, qL, H*S]
               tensor& kcache  // [B, H, kvLen, S]
) {
    auto B = q.size(0);
    auto qL = q.size(1);
    auto H = kcache.size(-3);
    auto kvLen = kcache.size(-2);
    auto S = kcache.size(-1);
    ASSERT(kcache.size(-4) == B);
    ASSERT((S % 8) == 0);

    q = q.reshape({B, qL, H, S}).permute({0, 2, 1, 3});  // B,H,qL,S

    float d_scale = 1.0f / std::sqrt(S);
    tensor attn_w;
    attn_w.reset(static_cast<float*>(nullptr), {B, H, qL, kvLen});

    auto qk_kernel_8x1 = [&](int64_t b, int64_t h, int64_t pq0, int64_t pq1, int64_t pk0, int64_t pk1) {
        auto* q_ptr = &q.at<float>({b, h, pq0, 0});
        auto q_stride = q.stride(2);
        auto q_stride3 = q.stride(2) * 3;
        auto* q_ptr2 = q_ptr + q_stride * 4;
        auto k_stride = kcache.stride(2);
        auto* k_ptr_base = &kcache.at<float>({b, h, pk0, 0});

        auto* pw_base = &attn_w.at<int32_t>({b, h, pq0, pk0});
        auto w_stride = attn_w.stride(2);

        auto v_d_scale = _mm256_broadcast_ss(&d_scale);

        for (int64_t pq = pq0; pq < pq1; pq += 8, q_ptr += 8 * q_stride, pw_base += 8 * w_stride) {
            auto* k_ptr0 = k_ptr_base;
            auto* pw0 = pw_base;
            for (int64_t pk = pk0; pk < pk1; pk += 1, k_ptr0 += k_stride, pw0++) {
                auto vsum0 = _mm256_setzero_ps();
                auto vsum1 = _mm256_setzero_ps();
                auto vsum2 = _mm256_setzero_ps();
                auto vsum3 = _mm256_setzero_ps();
                auto vsum4 = _mm256_setzero_ps();
                auto vsum5 = _mm256_setzero_ps();
                auto vsum6 = _mm256_setzero_ps();
                auto vsum7 = _mm256_setzero_ps();
                auto* k_ptr = k_ptr0;
                auto* q_ptr1 = q_ptr;
                auto* q_ptr2 = q_ptr1 + q_stride * 4;
                for (int64_t i = 0; i + 8 <= S; i += 8, q_ptr1 += 8, q_ptr2 += 8, k_ptr += 8) {
                    auto vb0 = mm256_uni_loadu_ps(k_ptr);

                    auto va0 = mm256_uni_loadu_ps(q_ptr1);
                    auto va1 = mm256_uni_loadu_ps(q_ptr1 + q_stride);
                    auto va2 = mm256_uni_loadu_ps(q_ptr1 + q_stride * 2);
                    auto va3 = mm256_uni_loadu_ps(q_ptr1 + q_stride3);

                    vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
                    vsum1 = _mm256_fmadd_ps(va1, vb0, vsum1);
                    vsum2 = _mm256_fmadd_ps(va2, vb0, vsum2);
                    vsum3 = _mm256_fmadd_ps(va3, vb0, vsum3);

                    auto va4 = mm256_uni_loadu_ps(q_ptr2);
                    auto va5 = mm256_uni_loadu_ps(q_ptr2 + q_stride);
                    auto va6 = mm256_uni_loadu_ps(q_ptr2 + q_stride * 2);
                    auto va7 = mm256_uni_loadu_ps(q_ptr2 + q_stride3);

                    vsum4 = _mm256_fmadd_ps(va4, vb0, vsum4);
                    vsum5 = _mm256_fmadd_ps(va5, vb0, vsum5);
                    vsum6 = _mm256_fmadd_ps(va6, vb0, vsum6);
                    vsum7 = _mm256_fmadd_ps(va7, vb0, vsum7);
                }

                transpose8_ps(vsum0, vsum1, vsum2, vsum3, vsum4, vsum5, vsum6, vsum7);
                vsum0 = _mm256_add_ps(vsum0, vsum1);
                vsum2 = _mm256_add_ps(vsum2, vsum3);
                vsum4 = _mm256_add_ps(vsum4, vsum5);
                vsum6 = _mm256_add_ps(vsum6, vsum7);
                vsum0 = _mm256_add_ps(vsum0, vsum2);
                vsum4 = _mm256_add_ps(vsum4, vsum6);
                vsum0 = _mm256_add_ps(vsum0, vsum4);
                vsum0 = _mm256_mul_ps(vsum0, v_d_scale);

                __m128 hi = _mm256_extractf128_ps(vsum0, 1);
                __m128 lo = _mm256_castps256_ps128(vsum0);

                auto* pw = pw0;
                pw[0] = _mm_extract_ps(lo, 0);
                pw += w_stride;
                pw[0] = _mm_extract_ps(lo, 1);
                pw += w_stride;
                pw[0] = _mm_extract_ps(lo, 2);
                pw += w_stride;
                pw[0] = _mm_extract_ps(lo, 3);
                pw += w_stride;
                pw[0] = _mm_extract_ps(hi, 0);
                pw += w_stride;
                pw[0] = _mm_extract_ps(hi, 1);
                pw += w_stride;
                pw[0] = _mm_extract_ps(hi, 2);
                pw += w_stride;
                pw[0] = _mm_extract_ps(hi, 3);
                pw += w_stride;
            }
        }
    };
    parallel_nt(0, B * H, 0, [&](int64_t bh0, int64_t bh1) {
        const int block_q = 64;
        const int block_k = 64;
        for (auto bh = bh0; bh < bh1; bh++) {
            auto h = bh % H;
            auto b = bh / H;
            for (int64_t pq0 = 0; pq0 < qL; pq0 += block_q) {
                auto pq1 = pq0 + block_q;
                if (pq1 > qL)
                    pq1 = qL;
                for (int64_t pk0 = 0; pk0 < kvLen; pk0 += block_k) {
                    auto pk1 = pk0 + block_k;
                    if (pk1 > kvLen)
                        pk1 = kvLen;
                    qk_kernel_8x1(b, h, pq0, pq1, pk0, pk1);
                }
            }
        }
    });
    return attn_w;
}

tensor mm_qk24(tensor& q,      // [B, qL, H*S]
               tensor& kcache  // [B, H, kvLen, S]
) {
    auto B = q.size(0);
    auto qL = q.size(1);
    auto H = kcache.size(-3);
    auto kvLen = kcache.size(-2);
    auto S = kcache.size(-1);
    ASSERT(kcache.size(-4) == B);
    ASSERT((S % 8) == 0);

    q = q.reshape({B, qL, H, S}).permute({0, 2, 1, 3});  // B,H,qL,S

    float d_scale = 1.0f / std::sqrt(S);
    tensor attn_w;
    attn_w.reset(static_cast<float*>(nullptr), {B, H, qL, kvLen});

    auto qk_kernel_2x4 = [&](int64_t b, int64_t h, int64_t pq0, int64_t pq1, int64_t pk0, int64_t pk1) {
        for (int64_t pq = pq0; pq < pq1; pq += 2) {
            for (int64_t pk = pk0; pk < pk1; pk += 4) {
                auto* q_ptr0 = &q.at<float>({b, h, pq, 0});
                auto* q_ptr1 = &q.at<float>({b, h, pq + 1, 0});

                auto* k_ptr = &kcache.at<float>({b, h, pk, 0});
                auto k_stride = kcache.stride(2);
                size_t i = 0;
                auto vsum0 = _mm256_setzero_ps();
                auto vsum1 = _mm256_setzero_ps();
                auto vsum2 = _mm256_setzero_ps();
                auto vsum3 = _mm256_setzero_ps();
                auto vsum4 = _mm256_setzero_ps();
                auto vsum5 = _mm256_setzero_ps();
                auto vsum6 = _mm256_setzero_ps();
                auto vsum7 = _mm256_setzero_ps();

                for (; i + 8 <= S; i += 8) {
                    auto vb0 = mm256_uni_loadu_ps(k_ptr + i);
                    auto vb1 = mm256_uni_loadu_ps(k_ptr + i + k_stride);
                    auto vb2 = mm256_uni_loadu_ps(k_ptr + i + k_stride * 2);
                    auto vb3 = mm256_uni_loadu_ps(k_ptr + i + k_stride * 3);

                    auto va0 = mm256_uni_loadu_ps(q_ptr0 + i);
                    auto va1 = mm256_uni_loadu_ps(q_ptr1 + i);

                    vsum0 = _mm256_fmadd_ps(vb0, va0, vsum0);
                    vsum1 = _mm256_fmadd_ps(vb1, va0, vsum1);
                    vsum2 = _mm256_fmadd_ps(vb2, va0, vsum2);
                    vsum3 = _mm256_fmadd_ps(vb3, va0, vsum3);

                    vsum4 = _mm256_fmadd_ps(vb0, va1, vsum4);
                    vsum5 = _mm256_fmadd_ps(vb1, va1, vsum5);
                    vsum6 = _mm256_fmadd_ps(vb2, va1, vsum6);
                    vsum7 = _mm256_fmadd_ps(vb3, va1, vsum7);
                }
                transpose8_ps(vsum0, vsum1, vsum2, vsum3, vsum4, vsum5, vsum6, vsum7);
                vsum0 = _mm256_add_ps(vsum0, vsum1);
                vsum2 = _mm256_add_ps(vsum2, vsum3);
                vsum4 = _mm256_add_ps(vsum4, vsum5);
                vsum6 = _mm256_add_ps(vsum6, vsum7);
                vsum0 = _mm256_add_ps(vsum0, vsum2);
                vsum4 = _mm256_add_ps(vsum4, vsum6);
                vsum0 = _mm256_add_ps(vsum0, vsum4);

                vsum0 = _mm256_mul_ps(vsum0, _mm256_broadcast_ss(&d_scale));
                auto* pw = &attn_w.at<float>({b, h, pq, pk});
                auto w_stride = attn_w.stride(2);
                _mm_storeu_ps(pw, _mm256_castps256_ps128(vsum0));
                pw += w_stride;
                _mm_storeu_ps(pw, _mm256_extractf128_ps(vsum0, 1));
            }
        }
    };
    parallel_nt(0, B * H, 0, [&](int64_t bh0, int64_t bh1) {
        constexpr int block = 128;
        for (auto bh = bh0; bh < bh1; bh++) {
            auto h = bh % H;
            auto b = bh / H;
            for (int64_t pq0 = 0; pq0 < qL; pq0 += block) {
                auto pq1 = pq0 + block;
                if (pq1 > qL)
                    pq1 = qL;
                for (int64_t pk0 = 0; pk0 < kvLen; pk0 += block) {
                    auto pk1 = pk0 + block;
                    if (pk1 > kvLen)
                        pk1 = kvLen;
                    qk_kernel_2x4(b, h, pq0, pq1, pk0, pk1);
                }
            }
        }
    });
    return attn_w;
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
    auto prof = PROFILE("0");
    // validate dtype & rank
    ASSERT(q.is<float>(3));
    ASSERT(k.is<float>(3));
    ASSERT(v.is<float>(3));
    ASSERT(kv_cache.is<float>(5));
    // std::cout << kvc_slots.repr() << std::endl;
    ASSERT(kvc_slots.is<int32_t>(1));

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

    prof = PROFILE("rope_qk");
    // put k/v into cache
    auto* slots = kvc_slots.data<int32_t>();
    auto kcache = kv_cache.slice(0, 2 * layer_id + 0, 2 * layer_id + 0);
    auto vcache = kv_cache.slice(0, 2 * layer_id + 1, 2 * layer_id + 1);
    rope_embed(q, inv_freq, position_id);
    rope_embed(k, inv_freq, position_id);
    prof = PROFILE("cpy_kv");
    // rope_embed_to(k, vcache, slots, inv_freq, position_id);
    parallel_nt(0, B * H * qL, 0, [&](int64_t bhl0, int64_t bhl1) {
        for (auto bhl = bhl0; bhl < bhl1; bhl++) {
            auto pk = bhl % qL;
            auto bh = (bhl / qL);
            auto h = bh % H;
            auto b = bh / H;
            memcpy(&kcache.at<float>({b, h, slots[pk], 0}), &k.at<float>({b, h, pk, 0}), sizeof(float) * S);
            memcpy(&vcache.at<float>({b, h, slots[pk], 0}), &v.at<float>({b, h, pk, 0}), sizeof(float) * S);
        }
    });

    auto kvLen = position_id + qL;
    if (kvLen > max_kv_length) {
        kvLen = max_kv_length;
    }

    prof = PROFILE("q@k'");
    // main attention logic
    auto d_scale = 1.0f / sqrt(S);
    tensor attn_w;
    attn_w.reset(static_cast<float*>(nullptr), {B, H, qL, kvLen});

    auto qk_kernel = [&](int64_t b, int64_t h, int64_t pq0, int64_t pq1, int64_t pk0, int64_t pk1) {
        for (int64_t pq = pq0; pq < pq1; pq++) {
            for (int64_t pk = pk0; pk < pk1; pk++) {
                float sum = 0;
                sum = dot_product(&q.at<float>({b, h, pq, 0}), &kcache.at<float>({b, h, pk, 0}), S);
                attn_w.at<float>({b, h, pq, pk}) = sum * d_scale;
            }
        }
    };

    static Env<int> optqk("OPTQK", 0);
    if (optqk.value == 3 && ((qL % 4) == 0) && ((kvLen % 2) == 0)) {
        mm_qk42(q, kcache, attn_w);
    } else if (optqk.value == 1) {
        parallel_nt(0, B * H, 0, [&](int64_t bh0, int64_t bh1) {
            for (auto bh = bh0; bh < bh1; bh++) {
                auto h = bh % H;
                auto b = bh / H;
                qk_kernel(b, h, 0, qL, 0, kvLen);
            }
        });
    } else if (optqk.value == 2) {
        parallel_nt(0, B * H, 0, [&](int64_t bh0, int64_t bh1) {
            constexpr int block = 128;
            for (auto bh = bh0; bh < bh1; bh++) {
                auto h = bh % H;
                auto b = bh / H;
                for (int64_t pq0 = 0; pq0 < qL; pq0 += block) {
                    auto pq1 = pq0 + block;
                    if (pq1 > qL)
                        pq1 = qL;
                    for (int64_t pk0 = 0; pk0 < kvLen; pk0 += block) {
                        auto pk1 = pk0 + block;
                        if (pk1 > kvLen)
                            pk1 = kvLen;
                        qk_kernel(b, h, pq0, pq1, pk0, pk1);
                    }
                }
            }
        });
    } else {
        parallel_nt(0, B * H * kvLen, 0, [&](int64_t bhl0, int64_t bhl1) {
            for (auto bhl = bhl0; bhl < bhl1; bhl++) {
                auto pk = bhl % kvLen;
                auto bh = (bhl / kvLen);
                auto h = bh % H;
                auto b = bh / H;
                for (int64_t pq = 0; pq < qL; pq++) {
                    float sum = 0;
                    sum = dot_product(&q.at<float>({b, h, pq, 0}), &kcache.at<float>({b, h, pk, 0}), S);
                    attn_w.at<float>({b, h, pq, pk}) = sum * d_scale;
                }
            }
        });
    }

    prof = PROFILE("softmax");
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

    prof = PROFILE("w*v");
    static Env<int> optwv("OPTWV", 0);
    if ((qL > 1 && optwv.value == 1) || (optwv.value == 2)) {
        parallel_nt(0, B * H, 0, [&](int64_t bh0, int64_t bh1) {
            for (auto bh = bh0; bh < bh1; bh++) {
                auto h = bh % H;
                auto b = bh / H;
                for (int64_t pq = 0; pq < qL; pq++) {
                    auto* dst = &q.at<float>({b, h, pq, 0});
                    memset(dst, 0, S * sizeof(dst[0]));
                    for (int64_t pv = 0; pv < kvLen; pv++) {
                        auto weight = attn_w.at<float>({b, h, pq, pv});
                        auto* v = &vcache.at<float>({b, h, pv, 0});
                        accumulate_weighted_v(dst, weight, v, S);
                    }
                }
            }
        });
        return;
    }

    tensor m_temp;
    auto nthr = omp_get_max_threads();
    m_temp.reset(static_cast<float*>(nullptr), {nthr, B, qL, H, S});

    int64_t work_amount = B * H * kvLen;
    parallel_nt(0, nthr, 0, [&](int64_t tid0, int64_t tid1) {
        auto ithr = omp_get_thread_num();
        memset(&m_temp.at<float>({ithr, 0, 0, 0, 0}), 0, m_temp.stride(0) * sizeof(float));

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

    prof = PROFILE("reduce");
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

// attention with RoPE and kv-cache
void attention_rope2(tensor q,          // [B, qL, H*S]
                     tensor k,          // [B, qL, H*S]
                     tensor v,          // [B, qL, H*S]
                     tensor inv_freq,   // [rotary_dims/2] for RoPE
                     tensor kv_cache,   // [2, B, H, max_length, S]
                     tensor kvc_slots,  // [qL]
                     int position_id,
                     int layer_id) {
    auto prof = PROFILE("0");
    // validate dtype & rank
    ASSERT(q.is<float>(3));
    ASSERT(k.is<float>(3));
    ASSERT(v.is<float>(3));
    ASSERT(kv_cache.is<float>(5));
    // std::cout << kvc_slots.repr() << std::endl;
    ASSERT(kvc_slots.is<int32_t>(1));

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

    prof = PROFILE("rope_qk");
    // put k/v into cache
    auto* slots = kvc_slots.data<int32_t>();
    auto kcache = kv_cache.slice(0, 2 * layer_id + 0, 2 * layer_id + 0);
    auto vcache = kv_cache.slice(0, 2 * layer_id + 1, 2 * layer_id + 1);
    rope_embed(q, inv_freq, position_id);
    rope_embed(k, inv_freq, position_id);
    prof = PROFILE("cpy_kv");
    // rope_embed_to(k, vcache, slots, inv_freq, position_id);
    parallel_nt(0, B * H * qL, 0, [&](int64_t bhl0, int64_t bhl1) {
        for (auto bhl = bhl0; bhl < bhl1; bhl++) {
            auto pk = bhl % qL;
            auto bh = (bhl / qL);
            auto h = bh % H;
            auto b = bh / H;
            memcpy(&kcache.at<float>({b, h, slots[pk], 0}), &k.at<float>({b, h, pk, 0}), sizeof(float) * S);
            memcpy(&vcache.at<float>({b, h, slots[pk], 0}), &v.at<float>({b, h, pk, 0}), sizeof(float) * S);
        }
    });

    auto kvLen = position_id + qL;
    if (kvLen > max_kv_length) {
        kvLen = max_kv_length;
    }

    // main attention logic
    prof = PROFILE("qkv");
    float d_scale = 1.0f / sqrt(S);

    auto qk_kernel_4x2 = [S, d_scale](float* q_ptr_base,
                                      int64_t q_stride,  // [q_len, S]
                                      float* k_ptr_base,
                                      int64_t k_stride,  // [k_len, S]
                                      int32_t* pw_base,
                                      int w_stride,  // [q_len, k_len]
                                      int64_t q_len,
                                      int64_t k_len) {
        auto q_stride1 = q_stride;
        auto q_stride3 = q_stride * 3;
        auto w_stride1 = w_stride;
        if (q_len < 4) {
            q_stride1 = 0;
            q_stride3 = 0;
            w_stride1 = 0;
        }
        for (int64_t pq = 0; pq < q_len;) {
            auto* k_ptr = k_ptr_base;

            for (int64_t pk = 0; pk < k_len; pk += 2, k_ptr += 2 * k_stride) {
                auto* q_ptr = q_ptr_base;
                auto* k_ptr0 = k_ptr;
                auto* k_ptr1 = k_ptr + k_stride;
                if (pk + 1 >= k_len) {
                    // k tails, prevent read overflow
                    k_ptr1 = k_ptr0;
                }
                auto vsum0 = _mm256_setzero_ps();
                auto vsum1 = _mm256_setzero_ps();
                auto vsum2 = _mm256_setzero_ps();
                auto vsum3 = _mm256_setzero_ps();
                auto vsum4 = _mm256_setzero_ps();
                auto vsum5 = _mm256_setzero_ps();
                auto vsum6 = _mm256_setzero_ps();
                auto vsum7 = _mm256_setzero_ps();

                for (int64_t i = 0; i + 8 <= S; i += 8, q_ptr += 8) {
                    auto va0 = mm256_uni_loadu_ps(q_ptr);
                    auto va1 = mm256_uni_loadu_ps(q_ptr + q_stride1);
                    auto va2 = mm256_uni_loadu_ps(q_ptr + q_stride1 * 2);
                    auto va3 = mm256_uni_loadu_ps(q_ptr + q_stride3);

                    auto vb0 = mm256_uni_loadu_ps(k_ptr0 + i);
                    vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
                    vsum1 = _mm256_fmadd_ps(va1, vb0, vsum1);
                    vsum2 = _mm256_fmadd_ps(va2, vb0, vsum2);
                    vsum3 = _mm256_fmadd_ps(va3, vb0, vsum3);

                    auto vb1 = mm256_uni_loadu_ps(k_ptr1 + i);
                    vsum4 = _mm256_fmadd_ps(va0, vb1, vsum4);
                    vsum5 = _mm256_fmadd_ps(va1, vb1, vsum5);
                    vsum6 = _mm256_fmadd_ps(va2, vb1, vsum6);
                    vsum7 = _mm256_fmadd_ps(va3, vb1, vsum7);
                }

                transpose8_ps(vsum0, vsum1, vsum2, vsum3, vsum4, vsum5, vsum6, vsum7);
                vsum0 = _mm256_add_ps(vsum0, vsum1);
                vsum2 = _mm256_add_ps(vsum2, vsum3);
                vsum4 = _mm256_add_ps(vsum4, vsum5);
                vsum6 = _mm256_add_ps(vsum6, vsum7);
                vsum0 = _mm256_add_ps(vsum0, vsum2);
                vsum4 = _mm256_add_ps(vsum4, vsum6);
                vsum0 = _mm256_add_ps(vsum0, vsum4);

                vsum0 = _mm256_mul_ps(vsum0, _mm256_broadcast_ss(&d_scale));
                __m128 hi = _mm256_extractf128_ps(vsum0, 1);
                __m128 lo = _mm256_castps256_ps128(vsum0);

                auto* pw = pw_base + pk;
                pw[0] = _mm_extract_ps(lo, 0);
                pw[w_stride1] = _mm_extract_ps(lo, 1);
                pw[w_stride1 * 2] = _mm_extract_ps(lo, 2);
                pw[w_stride1 * 3] = _mm_extract_ps(lo, 3);
                if (pk + 1 < k_len) {
                    // k tails, prevent write overflow
                    pw[1] = _mm_extract_ps(hi, 0);
                    pw[1 + w_stride1] = _mm_extract_ps(hi, 1);
                    pw[1 + w_stride1 * 2] = _mm_extract_ps(hi, 2);
                    pw[1 + w_stride1 * 3] = _mm_extract_ps(hi, 3);
                }
            }

            if (q_len < 4) {
                // total length in q is not enough to retreat
                // will compute in block size 1
                pq++;
                pw_base += w_stride;
                q_ptr_base += q_stride;
            } else {
                pq += 4;
                pw_base += 4 * w_stride;
                q_ptr_base += 4 * q_stride;
                // next round with step will overflow, retreat back (when q length is enough)
                if (pq + 4 > q_len) {
                    auto back_steps = (pq + 4) - q_len;
                    pw_base -= back_steps * w_stride;
                    q_ptr_base -= back_steps * q_stride;
                }
            }
        }
    };

    auto wv_kernel_4x16 = [S](float* w_ptr_base,
                              int64_t w_stride,  // [q_len, v_len]
                              float* v_ptr_base,
                              int64_t v_stride,  // [v_len, S]
                              float* d_ptr_vase,
                              int64_t d_stride,  // [q_len, S]
                              int64_t q_len,
                              int64_t v_len) {
        auto w_stride1 = w_stride;
        auto w_stride3 = w_stride * 3;
        auto d_stride1 = d_stride;
        if (q_len < 4) {
            // 4x16 kernel will act like 1x16 kernel
            w_stride1 = 0;
            w_stride3 = 0;
            d_stride1 = 0;
        }
        for (int64_t pq = 0; pq < q_len;) {
            for (int64_t s = 0; s < S; s += 16) {
                // 4x16 w*v
                auto* w_ptr1 = w_ptr_base;
                auto* v_ptr = v_ptr_base + s;
                auto vsum0 = _mm256_setzero_ps();
                auto vsum1 = _mm256_setzero_ps();
                auto vsum2 = _mm256_setzero_ps();
                auto vsum3 = _mm256_setzero_ps();
                auto vsum4 = _mm256_setzero_ps();
                auto vsum5 = _mm256_setzero_ps();
                auto vsum6 = _mm256_setzero_ps();
                auto vsum7 = _mm256_setzero_ps();
                for (int64_t pv = 0; pv < v_len; pv++, v_ptr += v_stride, w_ptr1++) {
                    auto v0 = mm256_uni_loadu_ps(v_ptr);
                    auto w0 = _mm256_broadcast_ss(w_ptr1);
                    auto w1 = _mm256_broadcast_ss(w_ptr1 + w_stride1);
                    auto w2 = _mm256_broadcast_ss(w_ptr1 + w_stride1 * 2);
                    auto w3 = _mm256_broadcast_ss(w_ptr1 + w_stride3);
                    vsum0 = _mm256_fmadd_ps(v0, w0, vsum0);
                    vsum1 = _mm256_fmadd_ps(v0, w1, vsum1);
                    vsum2 = _mm256_fmadd_ps(v0, w2, vsum2);
                    vsum3 = _mm256_fmadd_ps(v0, w3, vsum3);
                    auto v1 = mm256_uni_loadu_ps(v_ptr + 8);
                    vsum4 = _mm256_fmadd_ps(v1, w0, vsum4);
                    vsum5 = _mm256_fmadd_ps(v1, w1, vsum5);
                    vsum6 = _mm256_fmadd_ps(v1, w2, vsum6);
                    vsum7 = _mm256_fmadd_ps(v1, w3, vsum7);
                }
                auto* d_ptr = d_ptr_vase + s;
                mm256_uni_storeu_ps(d_ptr, vsum0);
                mm256_uni_storeu_ps(d_ptr + d_stride1, vsum1);
                mm256_uni_storeu_ps(d_ptr + d_stride1 * 2, vsum2);
                mm256_uni_storeu_ps(d_ptr + d_stride1 * 3, vsum3);
                mm256_uni_storeu_ps(d_ptr + 8, vsum4);
                mm256_uni_storeu_ps(d_ptr + d_stride1 + 8, vsum5);
                mm256_uni_storeu_ps(d_ptr + d_stride1 * 2 + 8, vsum6);
                mm256_uni_storeu_ps(d_ptr + d_stride1 * 3 + 8, vsum7);
            }
            if (q_len < 4) {
                // total length in q is not enough to retreat
                // will compute in block size 1
                pq++;
                w_ptr_base += w_stride;
                d_ptr_vase += d_stride;
            } else {
                pq += 4;
                w_ptr_base += 4 * w_stride;
                d_ptr_vase += 4 * d_stride;
                // next round with step will overflow, retreat back (when q length is enough)
                if (pq + 4 > q_len) {
                    auto back_steps = (pq + 4) - q_len;
                    w_ptr_base -= back_steps * w_stride;
                    d_ptr_vase -= back_steps * d_stride;
                }
            }
        }
    };

    const int block_q = 64;
    const int block_k = 64;
    ASSERT((S % 16) == 0);
    parallel_nt(0, B * H, 0, [&](int64_t bh0, int64_t bh1) {
        // per thread local buffer
        tensor attn_w;
        attn_w.reset(static_cast<float*>(nullptr), {block_q, kvLen});
        for (auto bh = bh0; bh < bh1; bh++) {
            auto h = bh % H;
            auto b = bh / H;
            for (int64_t pq0 = 0; pq0 < qL; pq0 += block_q) {
                auto pq1 = pq0 + block_q;
                if (pq1 > qL)
                    pq1 = qL;
                // Q*K'  [block_q, kvLen]
                prof = PROFILE("qk");
                for (int64_t pk0 = 0; pk0 < kvLen; pk0 += block_k) {
                    auto pk1 = pk0 + block_k;
                    if (pk1 > kvLen)
                        pk1 = kvLen;
                    
                    // check if the whole [block_q x block_k] block is casual masked
                    // 
                    for (int64_t pq = pq0; pq < pq1; pq++) {
                        auto* pw = &attn_w.at<float>({pq - pq0, 0});
                        for (int64_t pk = pq + 1; pk < qL; pk++) {
                            
                        }
                    }
                    qk_kernel_4x2(&q.at<float>({b, h, pq0, 0}),
                                  q.stride(2),
                                  &kcache.at<float>({b, h, pk0, 0}),
                                  kcache.stride(2),
                                  &attn_w.at<int32_t>({0, pk0}),
                                  kvLen,
                                  pq1 - pq0,
                                  pk1 - pk0);
                }
                prof = PROFILE("softmax");
                // block_q rows of attn is ready (& hot in cache), do softmax and W*V
                for (int64_t pq = pq0; pq < pq1; pq++) {
                    auto* pw = &attn_w.at<float>({pq - pq0, 0});
                    for (int64_t pk = pq + 1; pk < qL; pk++) {
                        pw[slots[pk]] = std::numeric_limits<float>::lowest();
                    }
                    _softmax(pw, kvLen);
                }
                prof = PROFILE("wv");
                // attn : [block_q, kvLen]
                // v    : [kvLen, S]
                // out  : [block_q, S]
                wv_kernel_4x16(&attn_w.at<float>({0, 0}),
                               attn_w.stride(0),
                               &vcache.at<float>({b, h, 0, 0}),
                               vcache.stride(2),
                               &q.at<float>({b, h, pq0, 0}),
                               q.stride(2),
                               pq1 - pq0,
                               kvLen);
            }
        }
    });
}
