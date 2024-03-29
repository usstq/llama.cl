#pragma once

#include <stdint.h>
#include "tensor.hpp"
#include "utils.hpp"

tensor clone(const tensor& old) {
  tensor newt;
  // allocate dense memory
  newt.reset(nullptr, *old.tinfo(), old.shape());
  // copy data
  auto d_rank = old.dense_rank();
  ASSERT(d_rank == old.m_rank);
  parallel_nt(0, old.numel(), 0, [&](int64_t i0, int64_t i1) {
    auto offset = i0 * old.m_item_size;
    auto sz = (i1 - i0) * old.m_item_size;
    memcpy(newt.data<int8_t>() + offset, old.data<int8_t>() + offset, sz);
  });
  return newt;
}

#if 0
#if defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
#include <sycl/sycl.hpp>
using namespace sycl;

static const int N = 8;

queue& get_sycl_queue() {
  static queue q;
  static auto qname = q.get_device().get_info<info::device::name>();
  std::cout << qname << std::endl;
  return q;
}

int syclmain() {
  auto& q = get_sycl_queue();
  int* data = malloc_shared<int>(N, q);
  for (int i = 0; i < N; i++)
    data[i] = i;

  q.parallel_for(range<1>(N), [=](id<1> i) { data[i] *= 2; }).wait();

  for (int i = 0; i < N; i++)
    std::cout << data[i] << ",";
  std::cout << std::endl;
  free(data, q);
  return 0;
}
#endif
#endif

// inplace add : a += b (with optinal broadcast)
// useful to represent bias/residual/...
void iadd(tensor a, tensor b) {
  ASSERT(a.is<float>());
  ASSERT(b.is<float>(1) || b.is<float>(a.rank()));  // rank-1 b is most useful
  ASSERT(b.size(-1) == a.size(-1));

  if (b.rank() == a.rank()) {
    ASSERT(b.shape() == a.shape());
  }

  // kernel on inner-most dimension
  auto inner_dim = a.size(-1);
  ASSERT((inner_dim % 8) == 0);
  parallel_nt(0, (a.numel() / inner_dim), 0, [&](int64_t i0, int64_t i1) {
    for (auto i = i0; i < i1; i++) {
      auto* dst = a.data<float>() + (i * inner_dim);
      auto* src = b.data<float>() + ((b.rank() == 1) ? 0 : (i * inner_dim));
      for (int64_t i = 0; i < inner_dim; i += 8) {
        auto v0 = _mm256_loadu_ps(src + i);
        auto v1 = _mm256_loadu_ps(dst + i);
        v1 = _mm256_add_ps(v1, v0);
        _mm256_storeu_ps(dst + i, v1);
      }
    }
  });
}

// inplace add : a += b (with optinal broadcast)
// useful to represent bias/residual/...
void imul(tensor a, tensor b) {
  ASSERT(a.is<float>());
  ASSERT(b.is<float>(1) || b.is<float>(a.rank()));  // rank-1 b is most useful
  ASSERT(b.size(-1) == a.size(-1));

  if (b.rank() == a.rank()) {
    ASSERT(b.shape() == a.shape());
  }

  // kernel on inner-most dimension
  auto inner_dim = a.size(-1);
  ASSERT((inner_dim % 8) == 0);
  parallel_nt(0, (a.numel() / inner_dim), 0, [&](int64_t i0, int64_t i1) {
    for (auto i = i0; i < i1; i++) {
      auto* dst = a.data<float>() + (i * inner_dim);
      auto* src = b.data<float>() + ((b.rank() == 1) ? 0 : (i * inner_dim));
      for (int64_t i = 0; i < inner_dim; i += 8) {
        auto v0 = _mm256_loadu_ps(src + i);
        auto v1 = _mm256_loadu_ps(dst + i);
        v1 = _mm256_mul_ps(v1, v0);
        _mm256_storeu_ps(dst + i, v1);
      }
    }
  });
}

template <typename F>
void _itran(tensor a, F kernel) {
  // kernel on inner-most dimension
  auto inner_dim = a.size(-1);
  ASSERT((inner_dim % 8) == 0);
  parallel_nt(0, (a.numel() / inner_dim), 0, [&](int64_t i0, int64_t i1) {
    for (auto i = i0; i < i1; i++) {
      auto* dst = a.data<float>() + (i * inner_dim);
      for (int64_t i = 0; i < inner_dim; i += 8) {
        auto v1 = _mm256_loadu_ps(dst + i);
        v1 = kernel(v1);
        _mm256_storeu_ps(dst + i, v1);
      }
    }
  });
}

// inplace transformation: activations/...
void itrans(tensor a, const std::string& op) {
  ASSERT(a.is<float>());
  if (op == "silu") {
    _itran(a, [](__m256 v) {
      // https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
      auto gate = sigmoid_avx2(v);
      return _mm256_mul_ps(v, gate);
    });
    return;
  }
  ASSERT(false);
}

// https://github.com/huggingface/transformers/blob/c5be38cd27bee92be73c73ba09aec8bedf841423/src/transformers/models/llama/modeling_llama.py#L105
// https://arxiv.org/pdf/1910.07467.pdf
void rmsnorm(tensor input, tensor weight, float variance_epsilon) {
  ASSERT(input.is<float>(3));   // [batch0, seq_len, hidden_states]
  ASSERT(weight.is<float>(1));  // [hidden_states]

  auto batch = input.size(0);
  auto seq_len = input.size(1);
  auto hidden_states = weight.size(0);
  ASSERT(input.size(-1) == hidden_states);
  ASSERT(hidden_states % 8 == 0);

  auto* wei = weight.data<float>();
  auto rank = input.rank();
  auto batch_size = input.numel() / hidden_states;
  parallel_nt(0, batch_size, 0, [&](int64_t bs0, int64_t bs1) {
    for (int64_t bs = bs0; bs < bs1; bs++) {
      auto b = (bs / seq_len);
      auto s = (bs % seq_len);
      auto* src = &input.at<float>({b, s, 0});

      // float rms = 0;
      auto vrms = _mm256_setzero_ps();
      for (int i = 0; i < hidden_states; i += 8) {
        // rms += src[i] * src[i];
        auto v0 = _mm256_loadu_ps(src + i);
        vrms = _mm256_fmadd_ps(v0, v0, vrms);
      }
      auto rms = _mm256_reduce_add_ps(vrms) / hidden_states;
      auto rsqrt_rms = _mm256_set1_ps(1.0f / std::sqrt(rms + variance_epsilon));

      for (int i = 0; i < hidden_states; i += 8) {
        // dst[i] = src[i] * rsqrt_rms * wei[i];
        auto v0 = _mm256_loadu_ps(src + i);
        auto w0 = _mm256_loadu_ps(wei + i);
        v0 = _mm256_mul_ps(v0, rsqrt_rms);
        v0 = _mm256_mul_ps(v0, w0);
        _mm256_storeu_ps(src + i, v0);
      }
    }
  });
}

void softmax(tensor input) {
  ASSERT(input.is<float>());  // [N, ..., K]
  ASSERT(input.is_dense());
  auto K = input.size(-1);
  auto batches = input.numel() / K;
  auto* psrc = input.data<float>();
  parallel_nt(0, batches, 0, [&](int64_t b0, int64_t b1) {
    for (int64_t b = b0; b < b1; b++) {
      _softmax(psrc + b * K, K);
    }
  });
}