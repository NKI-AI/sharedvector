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

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/offset_ptr.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>

#include "sharedvector/dtype.h"
#include "sharedvector/exceptions.h"

namespace bi = boost::interprocess;

namespace sharedvector {

// Lock-free atomics are address-free, which is what makes them valid inside a
// shared-memory segment mapped at different addresses in different processes.
static_assert(std::atomic<std::size_t>::is_always_lock_free,
              "sharedvector requires lock-free size_t atomics for cross-process use");

using ShmemMutex = bi::interprocess_mutex;
using ScopedLock = bi::scoped_lock<ShmemMutex>;

using ByteAllocator =
    bi::allocator<unsigned char, bi::managed_shared_memory::segment_manager>;
using SharedVectorData = bi::vector<unsigned char, ByteAllocator>;
using SizeTAllocator =
    bi::allocator<std::size_t, bi::managed_shared_memory::segment_manager>;
using SharedSizeVector = bi::vector<std::size_t, SizeTAllocator>;

struct SharedChunk {
  static constexpr std::size_t kMetaCapacity = 8;

  bi::offset_ptr<SharedVectorData> data;
  bi::offset_ptr<SharedSizeVector> shape;
  std::atomic<std::size_t> ref_count;
  ShmemMutex write_lock;
  double meta[kMetaCapacity];
  std::size_t meta_size;

  SharedChunk(bi::offset_ptr<SharedVectorData> d,
              bi::offset_ptr<SharedSizeVector> s)
      : data(d), shape(s), ref_count(0), meta{}, meta_size(0) {}

  SharedChunk(const SharedChunk&) = delete;
  SharedChunk& operator=(const SharedChunk&) = delete;
  SharedChunk(SharedChunk&&) = delete;
  SharedChunk& operator=(SharedChunk&&) = delete;

  template <typename T>
  const T* As() const {
    return reinterpret_cast<const T*>(data->data());
  }

  template <typename T>
  std::size_t Count() const {
    return data->size() / sizeof(T);
  }

  void SetMeta(const std::vector<double>& values) {
    std::copy(values.begin(), values.end(), meta);
    meta_size = values.size();
  }
};

struct SharedChunkDeleter {
  void operator()(SharedChunk* chunk) const {
    if (chunk) {
      chunk->ref_count.fetch_sub(1, std::memory_order_acq_rel);
    }
  }
};

using SharedChunkPtr = bi::offset_ptr<SharedChunk>;
using SharedChunkPtrAllocator =
    bi::allocator<SharedChunkPtr, bi::managed_shared_memory::segment_manager>;
using SharedChunksVector = bi::vector<SharedChunkPtr, SharedChunkPtrAllocator>;

struct SharedResources {
  bi::managed_shared_memory segment;
  ByteAllocator alloc_inst;
  SizeTAllocator size_t_alloc_inst;
  bi::allocator<SharedChunk, bi::managed_shared_memory::segment_manager>
      chunk_alloc_inst;
  ShmemMutex& mutex;
  std::atomic<std::size_t>* ref_count;
  // lock-free public size accessor
  std::atomic<std::size_t>* logical_size;

  // Opens the named segment; `create` selects open_or_create (owner) versus
  // open_only (attaching consumer), so consumers can never create a segment.
  static bi::managed_shared_memory OpenSegment(const std::string& name,
                                               std::size_t max_memory_size,
                                               bool create) {
    if (create) {
      return bi::managed_shared_memory(bi::open_or_create, name.c_str(),
                                       max_memory_size);
    }
    return bi::managed_shared_memory(bi::open_only, name.c_str());
  }

  SharedResources(const std::string& name, std::size_t max_memory_size,
                  bool create = true)
      : segment(OpenSegment(name, max_memory_size, create)),
        alloc_inst(segment.get_segment_manager()),
        size_t_alloc_inst(segment.get_segment_manager()),
        chunk_alloc_inst(segment.get_segment_manager()),
        mutex(*segment.find_or_construct<ShmemMutex>("mutex")()) {
    ref_count =
        segment.find_or_construct<std::atomic<std::size_t>>("ref_count")(0);
    logical_size = segment.find_or_construct<std::atomic<std::size_t>>(
        "logical_size")(0);
    ref_count->fetch_add(1, std::memory_order_acq_rel);
  }

  // Attached-process count; keeps the segment meaningful across independent
  // producer and consumer processes.
  ~SharedResources() { ref_count->fetch_sub(1, std::memory_order_acq_rel); }
};

class SharedVector {
 public:
  SharedVector(const std::string& name,
               std::size_t chunk_size = 1024 * 1024 * 1 /*1MB*/,
               std::size_t max_memory_size = 1024 * 1024 * 10,
               std::optional<DType> dtype = std::nullopt, bool create = true)
      : name_(name),
        chunk_size_(chunk_size),
        shared_resources_(std::make_unique<SharedResources>(
            name, max_memory_size, create)) {
    ScopedLock lock(shared_resources_->mutex);

    data_chunks_ =
        shared_resources_->segment.find<SharedChunksVector>("data_chunks")
            .first;
    if (!data_chunks_) {
      data_chunks_ = shared_resources_->segment.construct<SharedChunksVector>(
          "data_chunks")(shared_resources_->segment.get_segment_manager());
      // Only the creator sizes the chunk table; an attaching process passes
      // placeholder sizes and must not reallocate it inside a full segment.
      data_chunks_->reserve(max_memory_size / chunk_size_);
    }

    dtype_ = ResolveDType(dtype);

    shared_resources_->logical_size->store(data_chunks_->size(),
                                           std::memory_order_relaxed);
  }

  ~SharedVector() { ScopedLock lock(shared_resources_->mutex); }

  SharedResources& GetSharedResources() { return *shared_resources_; }

  SharedChunksVector* GetDataChunks() const { return data_chunks_; }

  DType GetDtype() const noexcept { return dtype_; }

  std::shared_ptr<SharedChunk> GetChunk(std::size_t index) {
    if (index >=
        shared_resources_->logical_size->load(std::memory_order_acquire)) {
      throw std::out_of_range("Index out of range");
    }

    SharedChunk* raw = (*data_chunks_)[index].get();

    {
      ScopedLock lock(raw->write_lock);
      raw->ref_count.fetch_add(1, std::memory_order_acq_rel);
    }

    return std::shared_ptr<SharedChunk>(raw, SharedChunkDeleter());
  }

  void AppendBytes(const void* bytes, std::size_t nbytes,
                   const std::vector<std::size_t>& shape,
                   const std::vector<double>& meta = {}) {
    if (nbytes > chunk_size_) {
      throw exceptions::MemoryError("Data size exceeds chunk capacity.");
    }
    ValidateMeta(meta);

    ScopedLock lock(shared_resources_->mutex);

    if (shared_resources_->segment.get_free_memory() <
        chunk_size_ + 256) {  // Add 256 as a small overhead
      throw exceptions::OutOfMemoryError("Not enough memory to append array.");
    }

    SharedVectorData* shared_data =
        shared_resources_->segment.construct<SharedVectorData>(
            bi::anonymous_instance)(shared_resources_->alloc_inst);

    const auto* begin = static_cast<const unsigned char*>(bytes);
    shared_data->reserve(chunk_size_);
    shared_data->assign(begin, begin + nbytes);

    SharedSizeVector* shared_shape =
        shared_resources_->segment.construct<SharedSizeVector>(
            bi::anonymous_instance)(shared_resources_->size_t_alloc_inst);
    shared_shape->assign(shape.begin(), shape.end());

    SharedChunk* new_chunk = shared_resources_->segment.construct<SharedChunk>(
        bi::anonymous_instance)(shared_data, shared_shape);
    new_chunk->SetMeta(meta);
    if (data_chunks_->size() >= data_chunks_->capacity()) {
      throw std::runtime_error("Data chunks vector is full");
    }
    data_chunks_->emplace_back(new_chunk);
    shared_resources_->logical_size->store(data_chunks_->size(),
                                           std::memory_order_release);
  }

  void ReplaceBytes(std::size_t index, const void* bytes, std::size_t nbytes,
                    const std::vector<std::size_t>& shape,
                    const std::vector<double>& meta = {}) {
    if (nbytes > chunk_size_) {
      throw exceptions::MemoryError("Array size exceeds chunk size");
    }
    ValidateMeta(meta);

    if (index >=
        shared_resources_->logical_size->load(std::memory_order_acquire)) {
      throw std::out_of_range("Index out of range");
    }

    SharedChunk* chunk = (*data_chunks_)[index].get();
    ScopedLock lock(chunk->write_lock);

    if (chunk->ref_count.load(std::memory_order_acquire) > 0) {
      throw std::runtime_error("Cannot replace chunk because it is in use");
    }

    const auto* begin = static_cast<const unsigned char*>(bytes);
    chunk->data->assign(begin, begin + nbytes);
    chunk->shape->assign(shape.begin(), shape.end());
    chunk->SetMeta(meta);
  }

  template <typename T>
  void append(const std::vector<T>& data, const std::vector<std::size_t>& shape,
              const std::vector<double>& meta = {}) {
    CheckDType(DTypeOf<T>::value);
    AppendBytes(data.data(), data.size() * sizeof(T), shape, meta);
  }

  template <typename T>
  void replace(std::size_t index, const std::vector<T>& data,
               const std::vector<std::size_t>& shape,
               const std::vector<double>& meta = {}) {
    CheckDType(DTypeOf<T>::value);
    ReplaceBytes(index, data.data(), data.size() * sizeof(T), shape, meta);
  }

  std::size_t size() const noexcept {
    return shared_resources_->logical_size->load(std::memory_order_acquire);
  }

  std::size_t GetChunkRefCount(std::size_t index) {
    if (index >=
        shared_resources_->logical_size->load(std::memory_order_acquire)) {
      throw std::out_of_range("Index out of range");
    }
    const SharedChunkPtr& chunk_ptr = (*data_chunks_)[index];
    if (!chunk_ptr) {
      throw std::runtime_error("Chunk pointer is null");
    }
    const SharedChunk& chunk = *chunk_ptr;
    return chunk.ref_count.load(std::memory_order_acquire);
  }

  std::size_t GetChunkPointer(std::size_t index) {
    if (index >=
        shared_resources_->logical_size->load(std::memory_order_acquire)) {
      throw std::out_of_range("Index out of range");
    }
    const SharedChunkPtr& chunk_ptr = (*data_chunks_)[index];
    if (!chunk_ptr || !chunk_ptr->data) {
      throw std::runtime_error("Chunk pointer or data is null");
    }
    return reinterpret_cast<std::size_t>(chunk_ptr->data->data());
  }

  std::vector<std::size_t> GetChunkShape(std::size_t index) const {
    if (index >=
        shared_resources_->logical_size->load(std::memory_order_acquire)) {
      throw std::out_of_range("Index out of range");
    }
    const SharedChunkPtr& chunk_ptr = (*data_chunks_)[index];
    if (!chunk_ptr || !chunk_ptr->shape) {
      throw std::runtime_error("Chunk pointer or shape is null");
    }
    const SharedSizeVector& shape = *chunk_ptr->shape;
    return std::vector<std::size_t>(shape.begin(), shape.end());
  }

  // Pins the chunk while copying, so the read cannot interleave with a replace.
  std::vector<double> GetChunkMeta(std::size_t index) {
    auto chunk = GetChunk(index);
    return std::vector<double>(chunk->meta, chunk->meta + chunk->meta_size);
  }

  std::size_t GetRefCount() const {
    return shared_resources_->ref_count->load(std::memory_order_acquire);
  }

  std::size_t GetFreeMemory() const {
    return shared_resources_->segment.get_free_memory();
  }

 private:
  // Adopts the dtype recorded in an existing segment or records the requested
  // one; the caller holds the segment mutex.
  DType ResolveDType(std::optional<DType> requested) {
    auto& segment = shared_resources_->segment;
    std::uint8_t* stored = segment.find<std::uint8_t>("dtype").first;
    if (stored) {
      const auto existing = static_cast<DType>(*stored);
      if (requested && *requested != existing) {
        throw std::invalid_argument(std::string("Segment dtype is ") +
                                    NumpyName(existing) + ", requested " +
                                    NumpyName(*requested));
      }
      return existing;
    }
    const DType chosen = requested.value_or(DType::kFloat32);
    segment.construct<std::uint8_t>("dtype")(
        static_cast<std::uint8_t>(chosen));
    return chosen;
  }

  void CheckDType(DType provided) const {
    if (provided != dtype_) {
      throw std::invalid_argument(std::string("Segment dtype is ") +
                                  NumpyName(dtype_) + ", data is " +
                                  NumpyName(provided));
    }
  }

  static void ValidateMeta(const std::vector<double>& meta) {
    if (meta.size() > SharedChunk::kMetaCapacity) {
      throw std::invalid_argument(
          "meta holds at most " + std::to_string(SharedChunk::kMetaCapacity) +
          " values, got " + std::to_string(meta.size()));
    }
  }

  std::string name_;
  std::size_t chunk_size_;
  std::unique_ptr<SharedResources> shared_resources_;
  SharedChunksVector* data_chunks_;
  DType dtype_ = DType::kFloat32;
};

}  // namespace sharedvector
