#pragma once

#include <stdint.h>

#include <vector>

#include "tensor.hpp"
#include "utils.hpp"

// https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html
template <typename Tidx, typename Tvalue = float>
static void _embedding_template(tensor output, tensor input, tensor weight) {
    auto embedding_dim = weight.size(1);
    auto* src = input.data<Tidx>();
    auto* dst = output.data<Tvalue>();
    parallel_nt(0, input.numel(), 0, [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; i++) {
            auto index = src[i];
            memcpy(dst + i * embedding_dim, &weight.at<Tvalue>({index, 0}), embedding_dim * weight.item_size());
        }
    });
}

void embedding(tensor output, tensor input, tensor weight) {
    ASSERT(weight.is<float>(2));
    ASSERT(output.is<float>());

    auto vocab_size = weight.size(0);
    auto embedding_dim = weight.size(1);
    auto rank = input.rank();

    auto infer_shape = [&]() {
        std::vector<int64_t> oshape(rank + 1);
        for (int i = 0; i < rank; i++)
            oshape[i] = input.size(i);
        oshape[rank] = embedding_dim;
        return oshape;
    };

    ASSERT(infer_shape() == output.shape());

    if (input.is<int32_t>()) {
        _embedding_template<int32_t>(output, input, weight);
    } else if (input.is<long>()) {
        _embedding_template<long>(output, input, weight);
    } else {
        _embedding_template<int64_t>(output, input, weight);
    }
}

std::vector<tensor> embedding_q8c_quant_w(tensor& input) {
    ASSERT(input.is<float>(2));
    auto B = input.size(0);
    auto K = input.size(1);

    std::vector<tensor> rets(2);
    auto& quantized = rets[0];
    auto& scales = rets[1];
    quantized.reset<int8_t>(nullptr, {B, K});
    scales.reset<float>(nullptr, {B});

    // kernel is light-weight to parallel, unless we have multiple rows
    parallel_nt(0, B, 0, [&](int64_t b0, int64_t b1) {
        for (auto b = b0; b < b1; b++) {
            float* raw_x = &input.at<float>({b, 0});
            int8_t* quant_x = &quantized.at<int8_t>({b, 0});
            auto amax = get_amax(raw_x, K);
            // x = (d * quantized)
            // quantized = round(x / d) = round(x * id)
            const float d = amax / 127;
            const float id = (d != 0) ? (1.0f / d) : 0;

            scales.at<float>({b}) = d;
            quant_row_q8_0(raw_x, quant_x, K, id);
        }
    });
    return rets;
}

template <typename Tidx>
static void _embedding_template_q8c(tensor output, tensor input, tensor weight, tensor scales) {
    auto embedding_dim = weight.size(1);
    auto* src0 = input.data<Tidx>();
    auto* dst0 = output.data<float>();
    parallel_nt(0, input.numel(), 0, [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; i++) {
            auto index = src0[i];
            auto* dst = dst0 + i * embedding_dim;
            // dequantize token @index
            auto* pw = &weight.at<int8_t>({index, 0});
            auto s = scales.at<float>({index});
            int64_t k = 0;
#if __AVX2__
            auto vs = _mm256_set1_ps(s);
            for (; k + 8 <= embedding_dim; k += 8) {
                auto ps = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadu_si64(pw + k)));
                ps = _mm256_mul_ps(ps, vs);
                _mm256_storeu_ps(dst + k, ps);
            }
#endif
            for (; k < embedding_dim; k++) {
                dst[k] = s * pw[k];
            }
        }
    });
}

void embedding_q8c(tensor output, tensor input, tensor weight, tensor scales) {
    ASSERT(weight.is<int8_t>(2));
    ASSERT(scales.is<float>(1));
    ASSERT(output.is<float>());

    auto vocab_size = weight.size(0);
    auto embedding_dim = weight.size(1);
    auto rank = input.rank();

    // input  : [D0, D1, ..., DN]
    // output : [D0, D1, ..., DN, E]   E=embedding_dim
    auto infer_shape = [&]() {
        std::vector<int64_t> oshape(rank + 1);
        for (int i = 0; i < rank; i++)
            oshape[i] = input.size(i);
        oshape[rank] = embedding_dim;
        return oshape;
    };

    ASSERT(infer_shape() == output.shape());

    if (input.is<int32_t>()) {
        _embedding_template_q8c<int32_t>(output, input, weight, scales);
    } else if (input.is<long>()) {
        _embedding_template_q8c<long>(output, input, weight, scales);
    } else {
        ASSERT(input.is<int64_t>());
        _embedding_template_q8c<int64_t>(output, input, weight, scales);
    }
}
