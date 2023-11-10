#include <torch/extension.h>

#include <iostream>
#include <sstream>
#include "intrinsic_helpers.hpp"
#include "ktensor.hpp"

#define INTEL_NO_ITTNOTIFY_API
#include <ittnotify.h>

// Torch tensor basics
//  https://pytorch.org/cppdocs/notes/tensor_indexing.html
//  include\ATen\core\TensorBody.h

#include <vector>

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

bool FC_dynamic_quantize_x(torch::Tensor &input,
                           torch::Tensor &x_quantized,
                           torch::Tensor &x_scales,
                           int64_t Kgroups, int64_t group_k,
                           float scale = 1.0f)
{
    __itt_event mark_event = __itt_event_create("DYNQ", 3);

    __itt_event_start(mark_event);

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
    __itt_event_end(mark_event);
    return true;
}

void kernel_evaluate_Q8C(Ktensor<int8_t> x_q8,    // B, M, Kgroups*group_k
                         Ktensor<float> x_scales, // B, M, Kgroups
                         Ktensor<float> output,   // B,M,N
                         Ktensor<q8_c_block> w_q8, // Ngroups, Kgroups
                         Ktensor<float> w_scales  // N
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

torch::Tensor FC_evaluate_Q8C(torch::Tensor input, torch::Tensor wei_quantized, torch::Tensor wei_scales, int N)
{
    static __itt_event mark_event = __itt_event_create("evalQ8C", 3);
    auto Ngroups = wei_quantized.size(0);
    auto Kgroups = wei_quantized.size(1);
    int group_k = 32;
    int group_n = 32;
    auto B = input.size(0);
    auto M = input.size(1);
    auto K = input.size(2);
    auto options =
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .layout(torch::kStrided)
            .device(torch::kCPU, 1)
            .requires_grad(false);

    auto output = torch::empty({B, M, N}, options);
    auto y_stride = output.stride(1);

    vnni_inst vnni_i8;

    torch::Tensor x_quantized;
    torch::Tensor x_scales;
    FC_dynamic_quantize_x(input, x_quantized, x_scales, Kgroups, group_k);

    __itt_event_start(mark_event);
    kernel_evaluate_Q8C(Ktensor<int8_t>(x_quantized.data_ptr<int8_t>(), {B, M, Kgroups*group_k}),    // B,M, Kgroups*group_k
                        Ktensor<float>(x_scales.data_ptr<float>(), {B, M, Kgroups}),
                        Ktensor<float>(output.data_ptr<float>(), {B, M, N}),
                        Ktensor<q8_c_block>(wei_quantized.data_ptr<int8_t>(), {Ngroups, Kgroups}),
                        Ktensor<float>(wei_scales.data_ptr<float>(), {N}));
    __itt_event_end(mark_event);

    return output;
}

float accTensor(torch::Tensor wei, int64_t i0, int64_t i1)
{
    at::parallel_for(0, 8, 1, [&](int64_t n, int64_t n1)
                     {
                        std::stringstream ss;
                        ss << n << std::endl;
                        std::cout << ss.str(); });
    return wei.index({i0, i1}).item<float>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("accTensor", &accTensor);
    m.def("FC_quant_Q8C", &FC_quant_Q8C, "FC_quant_Q8C");
    m.def("FC_evaluate_Q8C", &FC_evaluate_Q8C, "FC_evaluate_Q8C");
}