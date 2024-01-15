#pragma once

#include <stdint.h>

#include "tensor.hpp"
#include "utils.hpp"

//======================================================================================
// fullyconnect

// deq(q)=(d*q + m)

// K8 : how many 8 elements in K dimension groups together
template<int GROUP_K>
struct q4_1_block {
    static constexpr int group_k = GROUP_K;
    static constexpr int group_n = 32;
    static constexpr bool with_m = true;

    // whole (group_k x 32) block is stored in unit of 8x32, two 4x32 int4 are
    // stored in the high/low 4bits of one 4x32 int8 unit block.
    int8_t w[group_k/8][4 * 32];
    float16 wd[32];
    float16 wm[32];

    const float16* get_wd() const {
        return wd;
    }
    const float16* get_wm() const {
        return wm;
    }
    void set_d(int n, float16 d) {
        wd[n] = d;
    }
    void set_m(int n, float16 m) {
        wm[n] = m;
    }

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

template<int GROUP_K>
struct q4_0_block {
    static constexpr int group_k = GROUP_K;
    static constexpr int group_n = 32;
    static constexpr bool with_m = false;

    int8_t w[group_k/8][4 * 32];
    float16 wd[32];

    const float16* get_wd() const {
        return wd;
    }
    const float16* get_wm() const {
        return nullptr;
    }
    void set_d(int n, float16 d) {
        wd[n] = d;
    }
    void set_m(int n, float16 m) {
        ASSERT(m == 0);
    }
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
        return d * q;
    }
};

using q4_block = q4_1_block<32>;
//using q4_block = _q4_1_block<128>;

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
    int64_t group_k = q4_block::group_k;
    int64_t group_n = q4_block::group_n;
    ASSERT(weiq.size(2) == sizeof(q4_block));

    tensor output;
    output.reset<float>(nullptr, {Ngroups * group_n, Kgroups * group_k});

    parallel_nt(0, Ngroups, 0, [&](int64_t nb0, int64_t nb1) {
        for (auto nb = nb0; nb < nb1; nb++) {
            auto n0 = nb * group_n;
            q4_block* wq4 = &weiq.at<q4_block>({nb, 0, 0});
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
    int64_t group_k = q4_block::group_k;
    int64_t group_n = q4_block::group_n;
    int64_t Kgroups = (K + group_k - 1) / group_k;
    int64_t Ngroups = (N + group_n - 1) / group_n;

    tensor wei_quantized;
    wei_quantized.reset<int8_t>(nullptr, {Ngroups, Kgroups, static_cast<int64_t>(sizeof(q4_block))});

    // each 32x32 sub-block is further interleaved every 4-rows into (32/4)x(32*4)
    // and each column of 32x32 sub-block share a quantization scales
    parallel_nt(0, Ngroups, 0, [&](int64_t nb0, int64_t nb1) {
        for (auto nb = nb0; nb < nb1; nb++) {
            auto n0 = nb * group_n;
            q4_block* wq4 = &wei_quantized.at<q4_block>({nb, 0, 0});
            for (int64_t kb = 0, k0 = 0; kb < Kgroups; kb++, k0 += group_k, wq4++) {
                // w_q composed of
                for (int64_t ni = 0; ni < group_n; ni++) {
                    // asymmetric quantization
                    auto src_n = n0 + ni;
                    if (src_n >= N) {
                        wq4->set_d(ni, 0);
                        wq4->set_m(ni, 0);
                        continue;
                    }

                    float vmin, vmax, d, m, id;
                    float level_min, level_max;
                    get_min_max(&wei.at<float>({src_n, k0}), group_k, vmin, vmax);
                    if (q4_block::with_m) {
                        level_max = 15.0f;
                        level_min = 0.0f;
                        //  deq(q)=(d*q + m) maps [vmin,vmax] to [0, 15]
                        d = (vmax - vmin) / level_max;
                        m = vmin;
                        id = (d != 0) ? (1.0f / d) : 0;
                        wq4->set_d(ni, to_fp16(d));
                        wq4->set_m(ni, to_fp16(m));
                    } else {
                        level_max = 7.0f;
                        level_min = -7.0f;
                        //  deq(q)=(d*q) maps [-amax, amax] to [-7, 7]
                        float amax = std::max(std::abs(vmin), std::abs(vmax));
                        d = amax / level_max;
                        m = 0.0f;
                        id = (d != 0) ? (1.0f / d) : 0;
                        wq4->set_d(ni, to_fp16(d));
                    }

                    for (int ki = 0; ki < group_k; ki++) {
                        auto src_k = k0 + ki;
                        int8_t w_quantized = 0;
                        if (src_k < K) {
                            auto w_round = std::roundf((wei.at<float>({src_n, src_k}) - m) * id);
                            w_quantized = std::min(level_max, std::max(w_round, level_min));
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
            float* x_gsum = px_group_sum ? &(px_group_sum->at<float>({b, m, 0})) : nullptr;
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
              Wi ~ (Sw*Qwi + m)     // Qwi is 4bits unsigned

result      : sum[Sx*Qxi * (Sw*Qwi + m)]
              = (Sx*Sw) * sum(Qxi*Qwi) + m*sum(Sx*Qxi)
              = (Sx*Sw) * sum(Qxi*Qwi) + m*sum(Xi)

    sum(Qxi*Qwi) is calculated using AVX_VNNI
    sum(Sx*Qxi) is dynamically pre-calculated
*******************************************************************************/
tensor fc_Q4A(tensor input, tensor wei_quantized, int N) {
    ASSERT(input.is<float>(3));
    ASSERT(wei_quantized.is<int8_t>(3));
    auto Ngroups = wei_quantized.size(0);
    auto Kgroups = wei_quantized.size(1);
    int64_t group_k = q4_block::group_k;
    int64_t group_n = q4_block::group_n;
    auto B = input.size(0);
    auto M = input.size(1);
    auto K = input.size(2);

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

    // save dequantize scale for runtime to use
    // div 16 because de-compress 4bits into 8bits is easier when shifting
    // 4bits toward high-end of 8bits data, which imply multiplication of 16,
    // thus dequantize need to cancel this
    float additional_scale = q4_block::with_m ? 1.0f : (1.0f / 16.0f);
    FC_dynamic_quantize_x(input, x_quantized, x_scales, &x_group_sum, Kgroups, group_k, additional_scale);

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

                    const q4_block* wq4 = &wei_quantized.at<q4_block>({nb, 0, 0});
                    for (int kb = 0, k0 = 0; kb < Kgroups; kb++, k0 += group_k, q8_xd++, wq4++, xg_sum++) {
                        // K group is smallest quantization unit which shares single scale
                        auto acci0 = _mm256_setzero_si256();
                        auto acci1 = _mm256_setzero_si256();
                        auto acci2 = _mm256_setzero_si256();
                        auto acci3 = _mm256_setzero_si256();
                        const __m256i low4_mask = _mm256_set1_epi32(0x0F0F0F0F);
                        const __m256i high4_mask = _mm256_set1_epi32(0xF0F0F0F0);
                        auto* q4_weight = wq4->w[0];
                        for (int ki = 0; ki < group_k; ki += 8, q4_weight += 32 * 4, q8_xq += 8) {
                            // low 4bit 4x32 blocks
                            __m256i x0 = _mm256_set1_epi32(*reinterpret_cast<const int32_t*>(q8_xq));
                            __m256i y0 = _mm256_loadu_si256((const __m256i*)(q4_weight));
                            __m256i y1 = _mm256_loadu_si256((const __m256i*)(q4_weight + 32));
                            __m256i y2 = _mm256_loadu_si256((const __m256i*)(q4_weight + 32 * 2));
                            __m256i y3 = _mm256_loadu_si256((const __m256i*)(q4_weight + 32 * 3));

                            if (q4_block::with_m) {
                                acci0 = vnni(acci0, _mm256_and_si256(y0, low4_mask), x0);
                                acci1 = vnni(acci1, _mm256_and_si256(y1, low4_mask), x0);
                                acci2 = vnni(acci2, _mm256_and_si256(y2, low4_mask), x0);
                                acci3 = vnni(acci3, _mm256_and_si256(y3, low4_mask), x0);

                                // high 4bit
                                __m256i x1 = _mm256_set1_epi32(*reinterpret_cast<const int32_t*>(q8_xq + 4));
                                acci0 = vnni(acci0, _mm256_and_si256(_mm256_srli_epi16(y0, 4), low4_mask), x1);
                                acci1 = vnni(acci1, _mm256_and_si256(_mm256_srli_epi16(y1, 4), low4_mask), x1);
                                acci2 = vnni(acci2, _mm256_and_si256(_mm256_srli_epi16(y2, 4), low4_mask), x1);
                                acci3 = vnni(acci3, _mm256_and_si256(_mm256_srli_epi16(y3, 4), low4_mask), x1);
                            } else {
                                acci0 = vnni(acci0, _mm256_and_si256(_mm256_slli_epi16(y0, 4), high4_mask), x0);
                                acci1 = vnni(acci1, _mm256_and_si256(_mm256_slli_epi16(y1, 4), high4_mask), x0);
                                acci2 = vnni(acci2, _mm256_and_si256(_mm256_slli_epi16(y2, 4), high4_mask), x0);
                                acci3 = vnni(acci3, _mm256_and_si256(_mm256_slli_epi16(y3, 4), high4_mask), x0);

                                // high 4bit
                                __m256i x1 = _mm256_set1_epi32(*reinterpret_cast<const int32_t*>(q8_xq + 4));
                                acci0 = vnni(acci0, _mm256_and_si256(y0, high4_mask), x1);
                                acci1 = vnni(acci1, _mm256_and_si256(y1, high4_mask), x1);
                                acci2 = vnni(acci2, _mm256_and_si256(y2, high4_mask), x1);
                                acci3 = vnni(acci3, _mm256_and_si256(y3, high4_mask), x1);
                            }
                        }
                        // load de-quantize coeff and combine with input's dequantize coeff
                        // const u_int16_t *f16_scale = reinterpret_cast<const u_int16_t
                        // *>(&wei_scales({kb, n0}));
                        auto dx = _mm256_broadcast_ss(q8_xd);

                        auto d0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)wq4->wd));
                        auto d1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(&wq4->wd[8 * 1])));
                        auto d2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(&wq4->wd[8 * 2])));
                        auto d3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(&wq4->wd[8 * 3])));

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
                        if (q4_block::with_m) {
                            auto* wm = wq4->get_wm();
                            auto m0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)wm));
                            auto m1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(&wm[8 * 1])));
                            auto m2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(&wm[8 * 2])));
                            auto m3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(&wm[8 * 3])));

                            auto gsum = _mm256_broadcast_ss(xg_sum);
                            acc0 = _mm256_fmadd_ps(m0, gsum, acc0);
                            acc1 = _mm256_fmadd_ps(m1, gsum, acc1);
                            acc2 = _mm256_fmadd_ps(m2, gsum, acc2);
                            acc3 = _mm256_fmadd_ps(m3, gsum, acc3);
                        }
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

tensor fc_Q4A2(tensor input, tensor wei_quantized, int N) {
#if 0
    ASSERT(input.is<float>(3));
    ASSERT(wei_quantized.is<int8_t>(3));
    auto Ngroups = wei_quantized.size(0);
    auto Kgroups = wei_quantized.size(1);
    int64_t group_k = q4_block::group_k;
    int64_t group_n = q4_block::group_n;
    auto B = input.size(0);
    auto M = input.size(1);
    auto K = input.size(2);

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

    FC_dynamic_quantize_x(input, x_quantized, x_scales, &x_group_sum, Kgroups, group_k, 1.0f);

    auto BM = B * M;
    ASSERT(BM % 2 == 0);
    parallel_nt(0, Ngroups, 0, [&](int64_t nb0, int64_t nb1) {
        for (auto nb = nb0; nb < nb1; nb++) {
            auto n0 = nb * group_n;
            float* py = &output.at<float>({0, 0, n0});
            const float* q8_xd = &x_scales.at<float>({0, 0, 0});
            const float* xg_sum = &x_group_sum.at<float>({0, 0, 0});
            const int8_t* q8_xq = &x_quantized.at<int8_t>({0, 0, 0});

            // B & M dimensions are collapsed as 1 dimension
            for (int64_t bm = 0; bm < BM; bm += 2,
                            py += 2 * y_stride,
                            q8_xd += 2 * x_scales.stride(1),
                            xg_sum += 2 * x_group_sum.stride(1),
                            q8_xq += 2 * x_quantized.stride(1)) {

                auto* q8_xd0 = q8_xd;
                auto* xg_sum0 = xg_sum;
                auto* q8_xq0 = q8_xq;

                auto* q8_xd1 = q8_xd + x_scales.stride(1);
                auto* xg_sum1 = xg_sum + x_group_sum.stride(1);
                auto* q8_xq1 = q8_xq + x_quantized.stride(1);

                // 2x32 kernel
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();

                __m256 acc4 = _mm256_setzero_ps();
                __m256 acc5 = _mm256_setzero_ps();
                __m256 acc6 = _mm256_setzero_ps();
                __m256 acc7 = _mm256_setzero_ps();

                const q4_block* wq4 = &wei_quantized.at<q4_block>({nb, 0, 0});
                for (int kb = 0, k0 = 0; kb < Kgroups;
                        kb++, k0 += group_k, q8_xd0++, q8_xd1++, wq4++, xg_sum0++, xg_sum1++) {
                    // K group is smallest quantization unit which shares single scale
                    auto acci0 = _mm256_setzero_si256();
                    auto acci1 = _mm256_setzero_si256();
                    auto acci2 = _mm256_setzero_si256();
                    auto acci3 = _mm256_setzero_si256();
                    // second row
                    auto acci4 = _mm256_setzero_si256();
                    auto acci5 = _mm256_setzero_si256();
                    auto acci6 = _mm256_setzero_si256();
                    auto acci7 = _mm256_setzero_si256();

                    const __m256i low4_mask = _mm256_set1_epi32(0x0F0F0F0F);
                    auto* q4_weight = wq4->w[0];
                    for (int ki = 0; ki < group_k; ki += 8, q4_weight += 32 * 4, q8_xq0 += 8, q8_xq1 += 8) {
                        // low 4bit 4x32 blocks
                        __m256i x0 = _mm256_set1_epi32(*reinterpret_cast<const int32_t*>(q8_xq0));
                        __m256i x10 = _mm256_set1_epi32(*reinterpret_cast<const int32_t*>(q8_xq1));

                        __m256i y0 = _mm256_loadu_si256((const __m256i*)(q4_weight));
                        __m256i y1 = _mm256_loadu_si256((const __m256i*)(q4_weight + 32));
                        __m256i y2 = _mm256_loadu_si256((const __m256i*)(q4_weight + 32 * 2));
                        __m256i y3 = _mm256_loadu_si256((const __m256i*)(q4_weight + 32 * 3));

                        auto z0 = _mm256_and_si256(y0, low4_mask);
                        acci0 = vnni(acci0, z0, x0);
                        acci4 = vnni(acci4, z0, x10);

                        auto z1 = _mm256_and_si256(y1, low4_mask);
                        acci1 = vnni(acci1, z1, x0);
                        acci5 = vnni(acci5, z1, x10);
                        
                        auto z2 = _mm256_and_si256(y2, low4_mask);
                        acci2 = vnni(acci2, z2, x0);
                        acci6 = vnni(acci6, z2, x10);

                        auto z3 = _mm256_and_si256(y3, low4_mask);
                        acci3 = vnni(acci3, z3, x0);
                        acci7 = vnni(acci7, z3, x10);

                        // high 4bit
                        __m256i x1 = _mm256_set1_epi32(*reinterpret_cast<const int32_t*>(q8_xq0 + 4));
                        __m256i x11 = _mm256_set1_epi32(*reinterpret_cast<const int32_t*>(q8_xq1 + 4));
                        z0 = _mm256_and_si256(_mm256_srli_epi16(y0, 4), low4_mask);
                        acci0 = vnni(acci0, z0, x1);
                        acci4 = vnni(acci4, z0, x11);

                        z1 = _mm256_and_si256(_mm256_srli_epi16(y1, 4), low4_mask);
                        acci1 = vnni(acci1, z1, x1);
                        acci5 = vnni(acci5, z1, x11);
                        
                        z2 = _mm256_and_si256(_mm256_srli_epi16(y2, 4), low4_mask);
                        acci2 = vnni(acci2, z2, x1);
                        acci6 = vnni(acci6, z2, x11);

                        z3 = _mm256_and_si256(_mm256_srli_epi16(y3, 4), low4_mask);
                        acci3 = vnni(acci3, z3, x1);
                        acci7 = vnni(acci7, z3, x11);
                    }
                    // load de-quantize coeff and combine with input's dequantize coeff
                    // const u_int16_t *f16_scale = reinterpret_cast<const u_int16_t
                    // *>(&wei_scales({kb, n0}));
#if 0
                    auto dx = _mm256_broadcast_ss(q8_xd0);
                    auto dx1 = _mm256_broadcast_ss(q8_xd1);
                    acc0 = _mm256_fmadd_ps(dx, _mm256_cvtepi32_ps(acci0), acc0);
                    acc1 = _mm256_fmadd_ps(dx, _mm256_cvtepi32_ps(acci1), acc1);
                    acc2 = _mm256_fmadd_ps(dx, _mm256_cvtepi32_ps(acci2), acc2);
                    acc3 = _mm256_fmadd_ps(dx, _mm256_cvtepi32_ps(acci3), acc3);
                    acc4 = _mm256_fmadd_ps(dx, _mm256_cvtepi32_ps(acci4), acc4);
                    acc5 = _mm256_fmadd_ps(dx, _mm256_cvtepi32_ps(acci5), acc5);
                    acc6 = _mm256_fmadd_ps(dx, _mm256_cvtepi32_ps(acci6), acc6);
                    acc7 = _mm256_fmadd_ps(dx, _mm256_cvtepi32_ps(acci7), acc7);
                    
#else
                    auto dx = _mm256_broadcast_ss(q8_xd0);
                    auto dx1 = _mm256_broadcast_ss(q8_xd1);
                    auto d0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)wq4->wd));
                    auto d1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(&wq4->wd[8 * 1])));
                    auto d2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(&wq4->wd[8 * 2])));
                    auto d3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(&wq4->wd[8 * 3])));

                    // dequantize
                    acc0 = _mm256_fmadd_ps(_mm256_mul_ps(d0, dx), _mm256_cvtepi32_ps(acci0), acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_mul_ps(d1, dx), _mm256_cvtepi32_ps(acci1), acc1);
                    acc2 = _mm256_fmadd_ps(_mm256_mul_ps(d2, dx), _mm256_cvtepi32_ps(acci2), acc2);
                    acc3 = _mm256_fmadd_ps(_mm256_mul_ps(d3, dx), _mm256_cvtepi32_ps(acci3), acc3);

                    acc4 = _mm256_fmadd_ps(_mm256_mul_ps(d0, dx1), _mm256_cvtepi32_ps(acci4), acc4);
                    acc5 = _mm256_fmadd_ps(_mm256_mul_ps(d1, dx1), _mm256_cvtepi32_ps(acci5), acc5);
                    acc6 = _mm256_fmadd_ps(_mm256_mul_ps(d2, dx1), _mm256_cvtepi32_ps(acci6), acc6);
                    acc7 = _mm256_fmadd_ps(_mm256_mul_ps(d3, dx1), _mm256_cvtepi32_ps(acci7), acc7);

                    // compensation term caused by zero-points
                    auto m0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)wq4->wm));
                    auto m1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(&wq4->wm[8 * 1])));
                    auto m2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(&wq4->wm[8 * 2])));
                    auto m3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(&wq4->wm[8 * 3])));

                    auto gsum = _mm256_broadcast_ss(xg_sum0);
                    acc0 = _mm256_fmadd_ps(m0, gsum, acc0);
                    acc1 = _mm256_fmadd_ps(m1, gsum, acc1);
                    acc2 = _mm256_fmadd_ps(m2, gsum, acc2);
                    acc3 = _mm256_fmadd_ps(m3, gsum, acc3);

                    auto gsum1 = _mm256_broadcast_ss(xg_sum1);
                    acc4 = _mm256_fmadd_ps(m0, gsum1, acc4);
                    acc5 = _mm256_fmadd_ps(m1, gsum1, acc5);
                    acc6 = _mm256_fmadd_ps(m2, gsum1, acc6);
                    acc7 = _mm256_fmadd_ps(m3, gsum1, acc7);
#endif
                }

                // output 32 results
                _mm256_storeu_ps(py + 8 * 0, acc0);
                _mm256_storeu_ps(py + 8 * 1, acc1);
                _mm256_storeu_ps(py + 8 * 2, acc2);
                _mm256_storeu_ps(py + 8 * 3, acc3);

                auto* py1 = py + y_stride;
                _mm256_storeu_ps(py1 + 8 * 0, acc4);
                _mm256_storeu_ps(py1 + 8 * 1, acc5);
                _mm256_storeu_ps(py1 + 8 * 2, acc6);
                _mm256_storeu_ps(py1 + 8 * 3, acc7);
            }
        }
    });
    return output;
#endif
}
