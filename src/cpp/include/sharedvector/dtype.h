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

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sharedvector {

// Element type of a segment; the single source of truth for itemsize and numpy name.
enum class DType : std::uint8_t { kFloat32 = 0, kInt16 = 1 };

constexpr std::size_t ItemSize(DType dtype) {
  return dtype == DType::kFloat32 ? sizeof(float) : sizeof(std::int16_t);
}

constexpr const char* NumpyName(DType dtype) {
  return dtype == DType::kFloat32 ? "float32" : "int16";
}

inline DType ParseDType(const std::string& name) {
  if (name == "float32") return DType::kFloat32;
  if (name == "int16") return DType::kInt16;
  throw std::invalid_argument("Unsupported dtype: " + name +
                              " (expected float32 or int16)");
}

template <typename T>
struct DTypeOf;

template <>
struct DTypeOf<float> {
  static constexpr DType value = DType::kFloat32;
};

template <>
struct DTypeOf<std::int16_t> {
  static constexpr DType value = DType::kInt16;
};

}  // namespace sharedvector
