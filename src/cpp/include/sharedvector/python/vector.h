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
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "sharedvector/dtype.h"
#include "sharedvector/vector.h"

namespace py = pybind11;

namespace sharedvector::python {

// Keeps the owning vector and the pinned chunk alive for the lifetime of a view.
struct ArrayOwner {
  py::object owner;
  std::shared_ptr<SharedChunk> chunk_ptr;
};

inline py::array GetPython(py::object self, size_t index) {
  SharedVector& vector = self.cast<SharedVector&>();
  auto chunk_ptr = vector.GetChunk(index);
  const DType dtype = vector.GetDtype();

  std::vector<ssize_t> shape(chunk_ptr->shape->begin(), chunk_ptr->shape->end());
  std::vector<ssize_t> strides(shape.size());
  auto stride = static_cast<ssize_t>(ItemSize(dtype));
  for (ssize_t i = static_cast<ssize_t>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }

  auto* payload = new ArrayOwner{self, chunk_ptr};
  auto capsule = py::capsule(payload, [](void* ptr) {
    delete static_cast<ArrayOwner*>(ptr);
  });

  return py::array(py::dtype(NumpyName(dtype)), shape, strides,
                   chunk_ptr->data->data(), capsule);
}

inline py::buffer_info CheckedRequest(const SharedVector& vector,
                                      const py::array& array,
                                      const char* verb) {
  const py::dtype expected(NumpyName(vector.GetDtype()));
  if (array.dtype().num() != expected.num()) {
    throw std::invalid_argument(
        std::string("dtype mismatch: segment holds ") +
        NumpyName(vector.GetDtype()) + ", array is " +
        py::str(array.dtype()).cast<std::string>());
  }
  if (!(array.flags() & py::array::c_style)) {
    throw std::runtime_error(
        std::string("Array is not contiguous. Only contiguous arrays are "
                    "supported for ") +
        verb + ".");
  }
  return array.request();
}

inline std::vector<double> ToMeta(const py::object& meta) {
  if (meta.is_none()) {
    return {};
  }
  auto array =
      py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
          meta);
  if (!array || array.ndim() != 1) {
    throw std::invalid_argument("meta must be a 1-D float64 array");
  }
  return std::vector<double>(array.data(), array.data() + array.size());
}

inline void AppendPython(SharedVector& vector, py::array array,
                         py::object meta) {
  py::buffer_info info = CheckedRequest(vector, array, "appending");
  std::vector<size_t> shape(info.shape.begin(), info.shape.end());
  vector.AppendBytes(info.ptr,
                     static_cast<size_t>(info.size) *
                         static_cast<size_t>(info.itemsize),
                     shape, ToMeta(meta));
}

inline void ReplacePython(SharedVector& vector, size_t index, py::array array,
                          py::object meta) {
  py::buffer_info info = CheckedRequest(vector, array, "replacing");
  std::vector<size_t> shape(info.shape.begin(), info.shape.end());
  vector.ReplaceBytes(index, info.ptr,
                      static_cast<size_t>(info.size) *
                          static_cast<size_t>(info.itemsize),
                      shape, ToMeta(meta));
}

inline py::array_t<double> GetMetaPython(SharedVector& vector, size_t index) {
  std::vector<double> meta = vector.GetChunkMeta(index);
  return py::array_t<double>(static_cast<ssize_t>(meta.size()), meta.data());
}

template <typename T>
inline py::array_t<T> ToArray(const std::vector<T>& values) {
  return py::array_t<T>(static_cast<ssize_t>(values.size()), values.data());
}

inline py::array_t<std::uint64_t> ReadCountsPython(const SharedVector& vector) {
  return ToArray(vector.ReadCounts());
}

inline py::array_t<std::uint64_t> WriteCountsPython(const SharedVector& vector) {
  return ToArray(vector.WriteCounts());
}

inline py::array_t<std::uint64_t> LastReadsPython(const SharedVector& vector) {
  return ToArray(vector.LastReads());
}

inline py::array_t<std::uint32_t> ReadSamplesPython(const SharedVector& vector) {
  return ToArray(vector.ReadSamples());
}

inline py::dict StatsPython(const SharedVector& vector) {
  const StatsSnapshot s = vector.GetStatsSnapshot();
  py::dict out;
  out["replace_events"] = s.replace_events;
  out["reads_at_replace_total"] = s.reads_at_replace_total;
  out["reads_at_replace_max"] = s.reads_at_replace_max;
  out["blocked_replaces"] = s.blocked_replaces;
  out["sample_count"] = s.sample_count;
  out["sample_capacity"] = s.sample_capacity;
  return out;
}

}  // namespace sharedvector::python
