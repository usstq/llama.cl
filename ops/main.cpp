#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include "op_fc.hpp"
#include "op_misc.hpp"
#include "tensor.hpp"
#include "utils.hpp"

namespace py = pybind11;

tensor from_buffer(py::buffer b, bool copy = false) {
  py::buffer_info info = b.request();
  tensor ret;
  void* src_ptr = copy ? nullptr : info.ptr;
  if (info.format == py::format_descriptor<float>::format()) {
    ret.reset(reinterpret_cast<float*>(src_ptr), info.shape, info.strides);
  } else if (info.format == py::format_descriptor<int>::format()) {
    ret.reset(reinterpret_cast<int32_t*>(src_ptr), info.shape, info.strides);
  } else if (info.format == py::format_descriptor<long long>::format()) {
    ret.reset(reinterpret_cast<int64_t*>(src_ptr), info.shape, info.strides);
  } else if (info.format == py::format_descriptor<int8_t>::format()) {
    ret.reset(reinterpret_cast<int8_t*>(src_ptr), info.shape, info.strides);
  } else {
    std::stringstream ss;
    ss << "Incompatible format: " << info.format;
    throw std::runtime_error(ss.str());
  }
  if (copy) {
    ASSERT(ret.is_dense());
    memcpy(ret.data(), info.ptr, ret.byte_size());
  }
  return ret;
}

py::array to_numpy(tensor& p) {
  // https://stackoverflow.com/questions/44659924/returning-numpy-arrays-via-pybind11
  // Create a Python object that will free the allocated
  // memory when destroyed:
  auto* p_shr_ptr = new std::shared_ptr<void>(p.m_ptr);
  py::capsule free_when_done(p_shr_ptr, [](void* ptr) {
    delete reinterpret_cast<std::shared_ptr<void>*>(ptr);
  });
  if (p.is<float>())
    return py::array(p.shape<ssize_t>(), p.byte_strides<ssize_t>(),
                     p.data<float>(), free_when_done);
  if (p.is<int8_t>())
    return py::array(p.shape<ssize_t>(), p.byte_strides<ssize_t>(),
                     p.data<int8_t>(), free_when_done);
  return py::array(0, reinterpret_cast<int8_t*>(nullptr));
}

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
  m.def("ones", [](py::args args, const py::kwargs& kwargs) {
    std::vector<int64_t> shape;
    for (auto a : args) {
      shape.push_back(py::cast<int64_t>(a));
    }
    tensor ret;
    ret.reset<float>(nullptr, shape);
    ret = 1.0f;
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
      .def(py::init([](py::buffer b) { return from_buffer(b); }))
      .def_property_readonly("shape", &tensor::shape<int64_t>)
      .def_property_readonly("strides", &tensor::strides<int64_t>)
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
      .def("numel", &tensor::numel)
      .def("numpy", [](tensor& p) { return to_numpy(p); })
      .def(py::pickle(
          [](tensor& p) {
            // __getstate__ : Return a tuple that fully encodes the state
            return py::make_tuple(to_numpy(p));
          },
          [](py::tuple t) {  // __setstate__
            ASSERT(t.size() == 1);
            return from_buffer(t[0].cast<py::array>(), true);
          }))
      .def("__getitem__", [](tensor& t, py::tuple index) {
        // return a tensor view (even a single element indexing is also a
        // tensor)
        //
        return t;
      });

  m.def("embedding", &embedding);
  m.def("rmsnorm", &rmsnorm);
  m.def("iadd", &iadd);
  m.def("clone", &clone);

  m.def("offline_FC_quant_Q4A", &offline_FC_quant_Q4A);
  m.def("offline_FC_dequant_Q4A", &offline_FC_dequant_Q4A);
  m.def("fc_Q4A", &fc_Q4A);
  m.def(
      "subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");
}