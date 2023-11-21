#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include "tensor.hpp"

// https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html
tensor embedding(tensor input,  // [batch0, batch0, ... ]
                 tensor weight  // [vocab_size, embedding_dim]
) {
  assert(weight.is<float>());
  assert(input.is<int64_t>());
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
#pragma omp parallel for
  for (int64_t i = 0; i < input.size(); i++) {
    auto index = src[i];
    memcpy(dst + i * embedding_dim, &weight.at<float>({index, 0}),
           embedding_dim * weight.item_size());
  }
  return output;
}

/*
class OP_rms_norm:
    def __init__(self, weight, variance_epsilon) -> None:
        self.weight = torch.clone(weight)
        self.variance_epsilon = variance_epsilon
        pass

    def __call__(self, input):
        input_dtype = input.dtype
        input = input.to(torch.float32)
        variance = input.pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * input.to(input_dtype)

    def __repr__(self):
        return f"OP_rms_norm(weight: {self.weight.shape}{self.weight.dtype},
esp:{self.variance_epsilon})"
*/
tensor rmsnorm(tensor input,   // [batch0, seq_len, hidden_states]
               tensor weight,  // [hidden_states]
               float variance_epsilon) {
  assert(input.is<float>());
  assert(weight.is<float>());
  assert(input.rank() == 3);
  assert(weight.rank() == 1);
  auto batch = input.size(0);
  auto seq_len = input.size(1);
  auto hidden_states = weight.size(0);
  assert(input.size(-1) == hidden_states);

  tensor output;
  auto oshape = input.shape();
  output.reset<float>(nullptr, oshape);

  auto* wei = weight.data<float>();
  auto rank = input.rank();
  auto batch_size = input.size() / hidden_states;
#pragma omp parallel for
  for (int64_t bs = 0; bs < batch_size; bs++) {
    auto b = (bs / seq_len);
    auto s = (bs % seq_len);
    auto* src = &input.at<float>({b, s, 0});
    auto* dst = &output.at<float>({b, s, 0});
    float rms = 0;
    for (int i = 0; i < hidden_states; i++) {
      rms += src[i] * src[i];
    }
    rms = rms / hidden_states;
    auto rsqrt_rms = 1.0f / std::sqrt(rms + variance_epsilon);
    for (int i = 0; i < hidden_states; i++) {
      dst[i] = src[i] * rsqrt_rms * wei[i];
    }
  }
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