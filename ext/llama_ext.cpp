#include <torch/extension.h>

#include <iostream>
#include <sstream>
#include "intrinsic_helpers.hpp"
#include "ktensor.hpp"

#ifdef USE_ITT_API
#include <ittnotify.h>
__itt_domain *domain = __itt_domain_create("MyTraces.MyDomain");
static void itt_task_begin(const char *name)
{
    static __itt_string_handle *shMyTask = __itt_string_handle_create(name);
    __itt_task_begin(domain, __itt_null, __itt_null, shMyTask);
}
static void itt_task_end()
{
    __itt_task_end(domain);
}
#else
static void itt_task_begin(const char *name)
{
}
static void itt_task_end()
{
}
#endif

// Torch tensor basics
//  https://pytorch.org/cppdocs/notes/tensor_indexing.html
//  include\ATen\core\TensorBody.h

#include <vector>

template<typename T>
Ktensor<T> makeKtenor(torch::Tensor x, std::initializer_list<int64_t> sizes = {})
{
    if (sizes.size() > 0) {
         return Ktensor<T>(x.data_ptr(), sizes);
    }

    auto rank = x.dim();
    if (rank == 1) return Ktensor<T>(x.data_ptr(), {x.size(0)});
    if (rank == 2) return Ktensor<T>(x.data_ptr(), {x.size(0), x.size(1)});
    if (rank == 3) return Ktensor<T>(x.data_ptr(), {x.size(0), x.size(1), x.size(2)});
    if (rank == 4) return Ktensor<T>(x.data_ptr(), {x.size(0), x.size(1), x.size(2), x.size(3)});
    if (rank == 5) return Ktensor<T>(x.data_ptr(), {x.size(0), x.size(1), x.size(2), x.size(3), x.size(4)});
}

struct q8_c_block
{
    int8_t w[32 / 4][32 * 4];
    int8_t &at(int k, int n) { return w[k >> 2][(n * 4) + (k & 3)]; }
};

std::vector<at::Tensor> FC_quant_Q8C(torch::Tensor wei)
{
    auto N = wei.size(0);
    auto K = wei.size(1);
    constexpr auto group_k = 32;
    constexpr auto group_n = 32;
    auto Ngroups = (N + group_n - 1) / group_n;
    auto Kgroups = (K + group_k - 1) / group_k;
    auto options =
        torch::TensorOptions()
            .layout(torch::kStrided)
            .device(torch::kCPU, 1)
            .requires_grad(false);

    auto wei_quantized = torch::zeros({Ngroups, Kgroups, sizeof(q8_c_block)}, options.dtype(torch::kInt8));
    auto wei_scales = torch::zeros({Ngroups * group_n}, options.dtype(torch::kFloat32));

    at::parallel_for(0, Ngroups * group_n, 0, [&](int64_t n0, int64_t n1)
                     {
        for (auto n = n0; n < n1; n++) {
            if (n >= N) {
                wei_scales.index_put_({n}, 0);
                return;
            }

            /*
            float amax = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                auto a = std::abs(wei.index({n, k}).item<float>());
                if (amax < a)
                    amax = a;
            }*/
            float amax = get_amax(wei.index({n, 0}).data_ptr<float>(), K);
            // x = (d * q)
            // q = x / d = x * id
            float d = amax / 127;
            float id = (d != 0) ? (1.0f / d) : 0;

            wei_scales.index_put_({n}, d);

            // quantize column n
            auto nb = n / group_n;
            auto noff = (n - nb * group_n);
            q8_c_block *wq8 = reinterpret_cast<q8_c_block*>(wei_quantized.index({nb, 0, 0}).data_ptr<int8_t>());
            const float* pweight = wei.index({n, 0}).data_ptr<float>();
            for (int64_t kb = 0, k0 = 0; kb < Kgroups; kb++, k0 += group_k, wq8++) {
                for (int64_t ki = 0; ki < group_k; ki++) {
                    auto src_k = k0 + ki;
                    if (src_k < K) {
                        wq8->at(ki, noff) = std::roundf(pweight[src_k] * id);
                    } else {
                        wq8->at(ki, noff) = 0;
                    }
                }
            } 
            } });
    return {wei_quantized, wei_scales};
}


void FC_dynamic_quantize_x(Ktensor<float> &input,
                           Ktensor<int8_t> &x_quantized,
                           Ktensor<float> &x_scales,
                           Ktensor<float> * px_group_sum,
                           int64_t Kgroups,
                           int64_t group_k,
                           float scale) {
    auto B = input.size(0);
    auto M = input.size(1);
    auto K = input.size(2);
    // dynamically quantize whole inputs
    //x_quantized.resize({B, M, Kgroups * group_k});
    //x_scales.resize({B, M, Kgroups});
    //x_group_sum.resize({B, M, Kgroups});

    // kernel is light-weight to parallel, unless we have multiple rows
    at::parallel_for(0, B*M, 0, [&](int64_t bm0, int64_t bm1) {
        for (auto bm = bm0; bm < bm1; bm++) {
            auto b = bm / M;
            auto m = bm % M;
            // a single row quantized in K groups
            float *q8_xd = &x_scales({b, m, 0});
            float *x_gsum = px_group_sum ? &(*px_group_sum)({b, m, 0}) : nullptr;
            int8_t *q8_xq = &x_quantized({b, m, 0});
            float *raw_x = &input({b, m, 0});
            for (int64_t kb = 0, left_k = K; kb < Kgroups; kb++, raw_x += group_k, q8_xq += group_k, left_k -= group_k) {
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

#if 0
bool FC_dynamic_quantize_x(torch::Tensor &input,
                           torch::Tensor &x_quantized,
                           torch::Tensor &x_scales,
                           int64_t Kgroups, int64_t group_k,
                           float scale = 1.0f)
{
    itt_task_begin("DYNQ");

    auto B = input.size(0);
    auto M = input.size(1);
    auto K = input.size(2);

    auto options =
        torch::TensorOptions()
            .layout(torch::kStrided)
            .device(torch::kCPU, 1)
            .requires_grad(false);

    // dynamically quantize whole inputs
    x_quantized = torch::empty({B, M, Kgroups * group_k}, options.dtype(torch::kInt8));
    x_scales = torch::empty({B, M, Kgroups}, options.dtype(torch::kFloat32));

    // kernel is light-weight to parallel, unless we have multiple rows
    at::parallel_for(0, B * M, 1, [&](int64_t bm0, int64_t bm1)
                     {
        for (auto bm = bm0; bm < bm1; bm++) {
        int64_t b = bm / M;
        int64_t m = bm % M;
        // a single row quantized in K groups
        float *q8_xd = x_scales.index({b, m, 0}).data_ptr<float>();
        //float *x_gsum = x_group_sum.index({b, m, 0}).data_ptr<float>();
        int8_t *q8_xq = x_quantized.index({b, m, 0}).data_ptr<int8_t>();
        float *raw_x = input.index({b, m, 0}).data_ptr<float>();
        for (int64_t kb = 0, left_k = K; kb < Kgroups; kb++, raw_x += group_k, q8_xq += group_k, left_k -= group_k) {
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
            /*
            if (require_group_sum) {
                float group_sum = 0.0f;
                for (int ki = 0; ki < actual_len; ki++) {
                    group_sum += raw_x[ki];
                }
                x_gsum[kb] = group_sum;
            }
            */
        } } });
    itt_task_end();
    return true;
}
#endif

void kernel_evaluate_Q8C(Ktensor<int8_t> x_q8,     // B, M, Kgroups*group_k
                         Ktensor<float> x_scales,  // B, M, Kgroups
                         Ktensor<float> output,    // B,M,N
                         Ktensor<q8_c_block> w_q8, // Ngroups, Kgroups
                         Ktensor<float> w_scales   // N
)
{
    auto B = output.size(0);
    auto M = output.size(1);
    auto N = output.size(2);
    constexpr auto group_k = 32;
    constexpr auto group_n = 32;
    auto Ngroups = w_q8.size(0);
    auto Kgroups = w_q8.size(1);
    auto y_stride = output.stride(1);
    vnni_inst vnni_i8;

    at::parallel_for(0, Ngroups, 0, [&](int64_t nb0, int64_t nb1)
                     {
        for(auto nb = nb0; nb < nb1; nb++) {
        int64_t n0 = nb * group_n;
        float *pwei_scales = &w_scales({n0});
        // B & M dimensions are collapsed as 1 dimension
        for (int64_t b = 0; b < B; b++) {
            for (int64_t m = 0; m < M; m++) {
                float *py = &output({b, m, n0});
                const float *q8_xd = &x_scales({b, m, 0});
                const int8_t *q8_xq = &x_q8({b, m, 0});

                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();
                const __m256i ones = _mm256_set1_epi16(1);
                const q8_c_block *wq8 = &w_q8({nb, 0});
                //std::cout << std::hex << reinterpret_cast<const void*>(wq8) << std::endl;
                for (int kb = 0, k0 = 0; kb < Kgroups; kb++, k0 += group_k, q8_xd++, wq8++) {
                    // K group is smallest quantization unit which shares single scale
                    auto acci0 = _mm256_setzero_si256();
                    auto acci1 = _mm256_setzero_si256();
                    auto acci2 = _mm256_setzero_si256();
                    auto acci3 = _mm256_setzero_si256();
                    auto *q8_weight = wq8->w[0];
                    for (int ki = 0; ki < group_k; ki += 4, q8_weight += 32 * 4, q8_xq += 4) {
                        // 4x32 vnni kernel
                        __m256i x0 = _mm256_set1_epi32(*reinterpret_cast<const int32_t *>(q8_xq));
                        __m256i y0 = _mm256_loadu_si256((const __m256i *)(q8_weight));
                        __m256i y1 = _mm256_loadu_si256((const __m256i *)(q8_weight + 32));
                        __m256i y2 = _mm256_loadu_si256((const __m256i *)(q8_weight + 32 * 2));
                        __m256i y3 = _mm256_loadu_si256((const __m256i *)(q8_weight + 32 * 3));

                        // apply sign in x0 to y0~y3
                        y0 = _mm256_sign_epi8(y0, x0);
                        y1 = _mm256_sign_epi8(y1, x0);
                        y2 = _mm256_sign_epi8(y2, x0);
                        y3 = _mm256_sign_epi8(y3, x0);

                        // Get absolute values of x vectors (x becomes u8 : 0~128)
                        x0 = _mm256_sign_epi8(x0, x0);

                        // u8 x s8
                        y0 = _mm256_maddubs_epi16(x0, y0);
                        y1 = _mm256_maddubs_epi16(x0, y1);
                        y2 = _mm256_maddubs_epi16(x0, y2);
                        y3 = _mm256_maddubs_epi16(x0, y3);

                        acci0 = _mm256_add_epi32(acci0, _mm256_madd_epi16(y0, ones));
                        acci1 = _mm256_add_epi32(acci1, _mm256_madd_epi16(y1, ones));
                        acci2 = _mm256_add_epi32(acci2, _mm256_madd_epi16(y2, ones));
                        acci3 = _mm256_add_epi32(acci3, _mm256_madd_epi16(y3, ones));
                    }
                    auto dx = _mm256_broadcast_ss(q8_xd);
                    // dequantize per-group k
                    acc0 = _mm256_fmadd_ps(dx, _mm256_cvtepi32_ps(acci0), acc0);
                    acc1 = _mm256_fmadd_ps(dx, _mm256_cvtepi32_ps(acci1), acc1);
                    acc2 = _mm256_fmadd_ps(dx, _mm256_cvtepi32_ps(acci2), acc2);
                    acc3 = _mm256_fmadd_ps(dx, _mm256_cvtepi32_ps(acci3), acc3);
                }

                // dequant per-n (OC)
                auto d0 = _mm256_loadu_ps(pwei_scales);
                auto d1 = _mm256_loadu_ps(pwei_scales + 8);
                auto d2 = _mm256_loadu_ps(pwei_scales + 8 * 2);
                auto d3 = _mm256_loadu_ps(pwei_scales + 8 * 3);

                acc0 = _mm256_mul_ps(d0, acc0);
                acc1 = _mm256_mul_ps(d1, acc1);
                acc2 = _mm256_mul_ps(d2, acc2);
                acc3 = _mm256_mul_ps(d3, acc3);

                // output 32 results
                _mm256_storeu_ps(py + 8 * 0, acc0);
                _mm256_storeu_ps(py + 8 * 1, acc1);
                _mm256_storeu_ps(py + 8 * 2, acc2);
                _mm256_storeu_ps(py + 8 * 3, acc3);
            }
        } } });
}

torch::Tensor FC_evaluate_Q8C(torch::Tensor tinput, torch::Tensor twei_quantized, torch::Tensor twei_scales, int N)
{
    itt_task_begin("evalQ8C");
    auto Ngroups = twei_quantized.size(0);
    auto Kgroups = twei_quantized.size(1);
    int group_k = 32;
    int group_n = 32;
    auto B = tinput.size(0);
    auto M = tinput.size(1);
    auto K = tinput.size(2);
    auto options =
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .layout(torch::kStrided)
            .device(torch::kCPU, 1)
            .requires_grad(false);

    auto output = torch::empty({B, M, N}, options);
    auto y_stride = output.stride(1);

    vnni_inst vnni_i8;

    // dynamically quantize whole inputs
    auto tx_quantized = torch::empty({B, M, Kgroups * group_k}, options.dtype(torch::kInt8));
    auto tx_scales = torch::empty({B, M, Kgroups}, options.dtype(torch::kFloat32));
    auto x_quantized = makeKtenor<int8_t>(tx_quantized);
    auto x_scales = makeKtenor<float>(tx_scales);
    auto input = makeKtenor<float>(tinput);
    auto wei_quantized = makeKtenor<q8_c_block>(twei_quantized, {Ngroups, Kgroups});
    auto wei_scales = makeKtenor<float>(twei_scales, {N});
    FC_dynamic_quantize_x(input, x_quantized, x_scales, nullptr, Kgroups, group_k, 1.0f);

    kernel_evaluate_Q8C(x_quantized, // B,M, Kgroups*group_k
                        x_scales,
                        Ktensor<float>(output.data_ptr<float>(), {B, M, N}),
                        wei_quantized,
                        wei_scales);

    itt_task_end();
    return output;
}

typedef short float16;
float16 to_fp16(float v) {
    return static_cast<float16>(_mm_cvtsi128_si32(_mm256_cvtps_ph(_mm256_set1_ps(v), _MM_FROUND_CUR_DIRECTION)));
}

struct q4_1_block
{
    // whole 4-bit 32x32 block distributed as following
    //    8x(4x32) each 2 adjacent (4x32) is combined into low/high part of a 8bit 4x32
    int8_t w[4][32 * 4];
    float16 wd[32];
    float16 wm[32];

    void set(int k, int n, int8_t v)
    {
        // assert(v >= -8 && v < 7)
        auto &value = w[k >> 3][(n * 4) + (k & 3)];
        bool is_high_4bit = ((k / 4) & 1);
        if (is_high_4bit)
        {
            value = (value & 0x0F) | (v << 4);
        }
        else
        {
            value = (value & 0xF0) | (v & 0x0F);
        }
    }
};




torch::Tensor FC_quant_Q4A(torch::Tensor tweight)
{
    // raw weight input is NxK (transpose_b is true)
    // strides is decreasing, so inner-most dimension is at higher ranks
    int64_t N = tweight.size(0);
    int64_t K = tweight.size(1);
    int group_k = 32;
    int group_n = 32;
    int64_t Kgroups = (K + group_k - 1) / group_k;
    int64_t Ngroups = (N + group_n - 1) / group_n;

    auto options =
        torch::TensorOptions()
            .layout(torch::kStrided)
            .device(torch::kCPU, 1)
            .requires_grad(false);

    auto weight_quantized = torch::zeros({Ngroups, Kgroups, sizeof(q4_1_block)}, options.dtype(torch::kInt8));

    auto wei = makeKtenor<float>(tweight);
    auto wei_quantized = makeKtenor<q4_1_block>(weight_quantized, {Ngroups, Kgroups});

    // each 32x32 sub-block is further interleaved every 4-rows into (32/4)x(32*4)
    // and each column of 32x32 sub-block share a quantization scales
    at::parallel_for(0, Ngroups, 0, [&](int64_t nb0, int64_t nb1) {
        for (auto nb = nb0; nb < nb1; nb++) {
            auto n0 = nb * group_n;
            q4_1_block *wq4 = &wei_quantized({nb, 0});
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
                    get_min_max(&wei({src_n, k0}), group_k, vmin, vmax);

                    //  to use deq(q)=(d*q + m) to map (vmin,vmax) to 0-15
                    //     d = (vmax-vmin)/15
                    //     m = vmin
                    const float level_max = 15;
                    float d = (vmax - vmin) / level_max;
                    float m = vmin;
                    float id = (d != 0) ? (1.0f / d) : 0;

                    wq4->wd[ni] = to_fp16(d);
                    wq4->wm[ni] = to_fp16(m);

                    for (int ki = 0; ki < group_k; ki++) {
                        auto src_k = k0 + ki;
                        int8_t w_quantized = 0;
                        if (src_n < N && src_k < K) {
                            auto w_round = std::roundf((wei({src_n, src_k}) - m) * id);
                            w_quantized = std::min(level_max, std::max(w_round, 0.0f));
                        }
                        wq4->set(ki, ni, w_quantized);
                    }
                }
            }
        } });
    return weight_quantized;
}


/*****************************************************************************
target      : sum(Xi * Wi)
approximate : Xi ~ (Sx*Qxi)         // Qxi is 8bits signed
              Wi ~ (Sw*Qwi + m)     // Qwi is 2bits unsigned

result      : sum[Sx*Qxi * (Sw*Qwi + m)] = (Sx*Sw) * sum(Qxi*Qwi) + m * sum(Sx*Qxi)

    sum(Qxi*Qwi) is calculated using AVX_VNNI
    sum(Sx*Qxi) is dynamically pre-calculated
*******************************************************************************/
torch::Tensor FC_evaluate_Q4A(torch::Tensor tinput, torch::Tensor tweight, int N) {
    auto Ngroups = tweight.size(0);
    auto Kgroups = tweight.size(1);
    int group_k = 32;
    int group_n = 32;
    auto B = tinput.size(0);
    auto M = tinput.size(1);
    auto K = tinput.size(2);
    VNNI_Sequence vnni_raw;

    auto options =
        torch::TensorOptions()
            .layout(torch::kStrided)
            .device(torch::kCPU, 1)
            .requires_grad(false);

    auto toutput = torch::zeros({B, M, N}, options.dtype(torch::kFloat32));
    auto output = makeKtenor<float>(toutput);
    auto input = makeKtenor<float>(tinput);
    auto wei_quantized = makeKtenor<q4_1_block>(tweight, {Ngroups, Kgroups});

    auto y_stride = output.stride(1);

    // dynamically quantize whole inputs
    auto tx_quantized = torch::empty({B, M, Kgroups * group_k}, options.dtype(torch::kInt8));
    auto tx_scales = torch::empty({B, M, Kgroups}, options.dtype(torch::kFloat32));
    auto tx_group_sum = torch::empty({B, M, Kgroups}, options.dtype(torch::kFloat32));

    auto x_quantized = makeKtenor<int8_t>(tx_quantized);
    auto x_scales = makeKtenor<float>(tx_scales);
    auto x_group_sum = makeKtenor<float>(tx_group_sum);
    FC_dynamic_quantize_x(input, x_quantized, x_scales, &x_group_sum, Kgroups, group_k, 1.0f);

    at::parallel_for(0, Ngroups, 0, [&](int64_t nb0, int64_t nb1) {
        for (auto nb = nb0; nb < nb1; nb++) {            
            auto n0 = nb * group_n;
            float *py = &output({0, 0, n0});
            // B & M dimensions are collapsed as 1 dimension
            for (int64_t b = 0; b < B; b++) {
                for (int64_t m = 0; m < M; m++, py += y_stride) {
                    const float *q8_xd = &x_scales({b, m, 0});
                    const float *xg_sum = &x_group_sum({b, m, 0});
                    const int8_t *q8_xq = &x_quantized({b, m, 0});

                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();

                    const q4_1_block *wq4 = &wei_quantized({nb, 0});
                    for (int kb = 0, k0 = 0; kb < Kgroups; kb++, k0 += group_k, q8_xd++, wq4++, xg_sum++) {
                        // K group is smallest quantization unit which shares single scale
                        auto acci0 = _mm256_setzero_si256();
                        auto acci1 = _mm256_setzero_si256();
                        auto acci2 = _mm256_setzero_si256();
                        auto acci3 = _mm256_setzero_si256();
                        const __m256i low4_mask = _mm256_set1_epi32(0x0F0F0F0F);
                        auto *q4_weight = wq4->w[0];
                        for (int ki = 0; ki < group_k; ki += 8, q4_weight += 32 * 4, q8_xq += 8) {
                            // low 4bit 4x32 blocks
                            __m256i x0 = _mm256_set1_epi32(*reinterpret_cast<const int32_t *>(q8_xq));
                            __m256i y0 = _mm256_loadu_si256((const __m256i *)(q4_weight));
                            __m256i y1 = _mm256_loadu_si256((const __m256i *)(q4_weight + 32));
                            __m256i y2 = _mm256_loadu_si256((const __m256i *)(q4_weight + 32 * 2));
                            __m256i y3 = _mm256_loadu_si256((const __m256i *)(q4_weight + 32 * 3));

                            acci0 = vnni_raw(acci0, x0, _mm256_and_si256(y0, low4_mask));
                            acci1 = vnni_raw(acci1, x0, _mm256_and_si256(y1, low4_mask));
                            acci2 = vnni_raw(acci2, x0, _mm256_and_si256(y2, low4_mask));
                            acci3 = vnni_raw(acci3, x0, _mm256_and_si256(y3, low4_mask));

                            // high 4bit
                            __m256i x1 = _mm256_set1_epi32(*reinterpret_cast<const int32_t *>(q8_xq + 4));
                            acci0 = vnni_raw(acci0, x1, _mm256_and_si256(_mm256_srli_epi16(y0, 4), low4_mask));
                            acci1 = vnni_raw(acci1, x1, _mm256_and_si256(_mm256_srli_epi16(y1, 4), low4_mask));
                            acci2 = vnni_raw(acci2, x1, _mm256_and_si256(_mm256_srli_epi16(y2, 4), low4_mask));
                            acci3 = vnni_raw(acci3, x1, _mm256_and_si256(_mm256_srli_epi16(y3, 4), low4_mask));
                        }
                        // load de-quantize coeff and combine with input's dequantize coeff
                        // const u_int16_t *f16_scale = reinterpret_cast<const u_int16_t *>(&wei_scales({kb, n0}));
                        auto dx = _mm256_broadcast_ss(q8_xd);

                        auto d0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)wq4->wd));
                        auto d1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(&wq4->wd[8 * 1])));
                        auto d2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(&wq4->wd[8 * 2])));
                        auto d3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(&wq4->wd[8 * 3])));

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
                        auto m0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)wq4->wm));
                        auto m1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(&wq4->wm[8 * 1])));
                        auto m2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(&wq4->wm[8 * 2])));
                        auto m3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(&wq4->wm[8 * 3])));

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

    return toutput;
}

struct FCQ4A {
    torch::Tensor _weight;
    int _N;
    FCQ4A(torch::Tensor tweight) {
        _N = tweight.size(0);
        _weight = FC_quant_Q4A(tweight);
    }
    torch::Tensor forward(torch::Tensor input) {
        return FC_evaluate_Q4A(input, _weight, _N);
    }
    std::string to_string() const
    {
      std::ostringstream out;
      out << "FCQ4A(";
      out << ")";
      return out.str();
    }
};

using Ktensorf32 = Ktensor<float>;

void test(const Ktensorf32 & t) {
    std::cout << "test" << t << std::endl;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("FC_quant_Q8C", &FC_quant_Q8C, "FC_quant_Q8C");
    m.def("FC_evaluate_Q8C", &FC_evaluate_Q8C, "FC_evaluate_Q8C");

    m.def("FC_quant_Q4A", &FC_quant_Q4A, "FC_quant_Q4A");
    m.def("FC_evaluate_Q4A", &FC_evaluate_Q4A, "FC_evaluate_Q4A");

    py::class_<Ktensorf32>(m, "Ktensorf32")
        .def(py::init<at::Tensor>());

    m.def("test", &test);
    //m.def("test", [](torch::Tensor t){
    //    test(t);
    //});
    py::class_<FCQ4A>(m, "FCQ4A", py::dynamic_attr())
        .def(py::init<torch::Tensor>())
        .def_readwrite("_weight", &FCQ4A::_weight)
        .def_readwrite("_N", &FCQ4A::_N)
        .def("forward", &FCQ4A::forward)
        .def("__repr__", &FCQ4A::to_string);
}