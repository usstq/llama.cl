#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include "tensor.hpp"

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

static inline float _mm256_reduce_add_ps(const __m256 x) {
  __m128 res = _mm256_extractf128_ps(x, 1);
  res = _mm_add_ps(res, _mm256_castps256_ps128(x));
  res = _mm_add_ps(res, _mm_movehl_ps(res, res));
  res = _mm_add_ss(res, _mm_movehdup_ps(res));
  return _mm_cvtss_f32(res);
}

#include <omp.h>
inline int64_t divup(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

template <typename F>
inline void parallel_nt(int64_t begin,
                        int64_t end,
                        int64_t grain_size,
                        const F& f) {
#pragma omp parallel
  {
    // choose number of tasks based on grain size and number of threads
    // can't use num_threads clause due to bugs in GOMP's thread pool (See
    // #32008)
    int64_t num_threads = omp_get_num_threads();
    if (grain_size > 0) {
      num_threads = std::min(num_threads, divup((end - begin), grain_size));
    }

    int64_t tid = omp_get_thread_num();
    int64_t chunk_size = divup((end - begin), num_threads);
    int64_t begin_tid = begin + tid * chunk_size;
    if (begin_tid < end) {
      f(begin_tid, std::min(end, chunk_size + begin_tid));
    }
  }
}

// https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html
tensor embedding(tensor input, tensor weight) {
  ASSERT(input.is<int64_t>());
  ASSERT(weight.is<float>(2));

  auto vocab_size = weight.size(0);
  auto embedding_dim = weight.size(1);

  auto rank = input.rank();
  std::vector<int64_t> oshape(rank + 1);

  for (int i = 0; i < rank; i++)
    oshape[i] = input.size(i);
  oshape[rank] = embedding_dim;

  tensor output;
  output.reset<float>(nullptr, oshape);
  auto* src = input.data<int64_t>();
  auto* dst = output.data<float>();
  parallel_nt(0, input.size(), 0, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; i++) {
      auto index = src[i];
      memcpy(dst + i * embedding_dim, &weight.at<float>({index, 0}),
             embedding_dim * weight.item_size());
    }
  });
  return output;
}

// https://github.com/huggingface/transformers/blob/c5be38cd27bee92be73c73ba09aec8bedf841423/src/transformers/models/llama/modeling_llama.py#L105
// https://arxiv.org/pdf/1910.07467.pdf
tensor rmsnorm(tensor input, tensor weight, float variance_epsilon) {
  ASSERT(input.is<float>(3));   // [batch0, seq_len, hidden_states]
  ASSERT(weight.is<float>(1));  // [hidden_states]

  auto batch = input.size(0);
  auto seq_len = input.size(1);
  auto hidden_states = weight.size(0);
  ASSERT(input.size(-1) == hidden_states);
  ASSERT(hidden_states % 8 == 0);

  tensor output;
  auto oshape = input.shape();
  output.reset<float>(nullptr, oshape);

  auto* wei = weight.data<float>();
  auto rank = input.rank();
  auto batch_size = input.size() / hidden_states;
  parallel_nt(0, batch_size, 0, [&](int64_t bs0, int64_t bs1) {
    for (int64_t bs = bs0; bs < bs1; bs++) {
      auto b = (bs / seq_len);
      auto s = (bs % seq_len);
      auto* src = &input.at<float>({b, s, 0});
      auto* dst = &output.at<float>({b, s, 0});
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
        _mm256_storeu_ps(dst + i, v0);
      }
    }
  });
  return output;
}

namespace py = pybind11;

PYBIND11_MODULE(llmops, m) {
  m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

  m.def("empty", [](py::args args, const py::kwargs& kwargs) {
    std::vector<int64_t> shape;
    for (auto a : args) {
      shape.push_back(py::cast<int64_t>(a));
    }
    tensor ret;
    ret.reset<float>(nullptr, shape);
    return ret;
  });

  py::class_<tensor>(m, "tensor", py::buffer_protocol())
      .def_buffer([](tensor& t) -> py::buffer_info {
        std::vector<ssize_t> shape(t.m_rank);
        std::vector<ssize_t> strides_in_bytes(t.m_rank);
        for (int i = 0; i < t.m_rank; i++) {
          shape[i] = t.size(i);
          strides_in_bytes[i] = t.stride(i) * t.item_size();
        }

        return py::buffer_info(
            t.data(),        /* Pointer to buffer */
            t.item_size(),   /* Size of one scalar */
            t.m_format,      /* Python struct-style format descriptor */
            t.m_rank,        /* Number of dimensions */
            shape,           /* Buffer dimensions */
            strides_in_bytes /* Strides (in bytes) for each index */
        );
      })
      .def(py::init([](py::buffer b) {
        py::buffer_info info = b.request();
        tensor ret;
        if (info.format == py::format_descriptor<float>::format()) {
          ret.reset(reinterpret_cast<float*>(info.ptr), info.shape,
                    info.strides);
        } else if (info.format == py::format_descriptor<int>::format()) {
          ret.reset(reinterpret_cast<int32_t*>(info.ptr), info.shape,
                    info.strides);
        } else if (info.format == py::format_descriptor<long long>::format()) {
          ret.reset(reinterpret_cast<int64_t*>(info.ptr), info.shape,
                    info.strides);
        } else {
          std::stringstream ss;
          ss << "Incompatible format: " << info.format;
          throw std::runtime_error(ss.str());
        }
        return ret;
      }))
      .def_property_readonly("shape", &tensor::shape)
      .def_property_readonly("strides", &tensor::strides)
      .def_property_readonly("item_size", &tensor::item_size)
      .def_property_readonly(
          "data",
          [](tensor& t) { return reinterpret_cast<std::uintptr_t>(t.data()); })
      .def("__str__",
           [](tensor& t) {
             std::stringstream ss;
             ss << t;
             return ss.str();
           })
      .def("clone", &tensor::clone)
      .def("__getitem__", [](tensor& t, py::tuple index) {
        // return a tensor view (even a single element indexing is also a
        // tensor)
        //
        return t;
      });

  m.def("embedding", &embedding);
  m.def("rmsnorm", &rmsnorm);
  m.def(
      "subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");
}