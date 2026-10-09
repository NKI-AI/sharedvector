// Copyright 2024 Jonas Teuwen. All Rights Reserved.
// Copyright 2026 AI for Oncology. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "sharedvector/dtype.h"
#include "sharedvector/exceptions.h"
#include "sharedvector/python/vector.h"
#include "sharedvector/vector.h"

namespace py = pybind11;
using sharedvector::DType;
using sharedvector::NumpyName;
using sharedvector::ParseDType;
using sharedvector::SharedVector;
using sharedvector::exceptions::ArraySizeError;
using sharedvector::exceptions::DataModifiedError;
using sharedvector::exceptions::InUseError;
using sharedvector::exceptions::MemoryError;
using sharedvector::exceptions::OutOfMemoryError;
namespace bi = boost::interprocess;

PYBIND11_MODULE(_vector, m) {
  py::class_<SharedVector>(m, "SharedVector")
      .def(py::init([](const std::string& name, size_t chunk_size,
                       size_t max_memory_size,
                       std::optional<std::string> dtype, bool create) {
             std::optional<DType> parsed;
             if (dtype) {
               parsed = ParseDType(*dtype);
             }
             return std::make_unique<SharedVector>(
                 name, chunk_size, max_memory_size, parsed, create);
           }),
           py::arg("name"), py::arg("chunk_size") = 1024 * 1024,
           py::arg("max_memory_size") = 1024 * 1024 * 10,
           py::arg("dtype") = py::none(), py::arg("create") = true)
      .def("append", &sharedvector::python::AppendPython,
           py::arg("array"), py::arg("meta") = py::none())
      .def("get", &sharedvector::python::GetPython, py::arg("index"))
      .def("get_meta", &sharedvector::python::GetMetaPython,
           py::arg("index"))
      .def("replace", &sharedvector::python::ReplacePython,
           py::arg("index"), py::arg("array"), py::arg("meta") = py::none())
      .def("size", &SharedVector::size)
      .def("__len__", &SharedVector::size)
      .def("get_chunk_ref_count", &SharedVector::GetChunkRefCount,
           py::arg("index"))
      .def(
          "get_chunk_shape",
          [](const SharedVector& self, size_t index) {
            std::vector<size_t> vec = self.GetChunkShape(index);
            py::tuple result = py::cast(vec);
            return result;
          },
          "Get the shape of a chunk as a tuple")
      .def("get_chunk_pointer", &SharedVector::GetChunkPointer,
           py::arg("index"))
      .def_property_readonly("free_memory", &SharedVector::GetFreeMemory)
      .def_property_readonly("ref_count", &SharedVector::GetRefCount)
      .def_property_readonly("dtype", [](const SharedVector& self) {
        return std::string(NumpyName(self.GetDtype()));
      });

  m.def("remove_shared_memory", [](const std::string& name) {
    bi::shared_memory_object::remove(name.c_str());
  });

  // A missing segment on an open_only attach surfaces as FileNotFoundError;
  // every other Boost.Interprocess failure stays a RuntimeError.
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const bi::interprocess_exception& e) {
      if (e.get_error_code() == bi::not_found_error) {
        PyErr_SetString(PyExc_FileNotFoundError, e.what());
      } else {
        PyErr_SetString(PyExc_RuntimeError, e.what());
      }
    }
  });

  py::register_exception<MemoryError>(m, "MemoryError");
  py::register_exception<OutOfMemoryError>(m, "OutOfMemoryError");
  py::register_exception<InUseError>(m, "InUseError");
  py::register_exception<ArraySizeError>(m, "ArraySizeError");
  py::register_exception<DataModifiedError>(m, "DataModifiedError");
}
