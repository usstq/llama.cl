#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include "op_fc.hpp"
#include "op_misc.hpp"
#include "tensor.hpp"
#include "utils.hpp"

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
      .def("numel", &tensor::numel)
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