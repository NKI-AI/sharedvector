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
#include <climits>
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
  // Read statistics: reads since the last write, number of writes (the append
  // counts as the first), and the read count captured by the last replace.
  std::atomic<std::uint64_t> read_count;
  std::atomic<std::uint64_t> write_count;
  std::atomic<std::uint64_t> last_reads;

  SharedChunk(bi::offset_ptr<SharedVectorData> d,
              bi::offset_ptr<SharedSizeVector> s)
      : data(d),
        shape(s),
        ref_count(0),
        meta{},
        meta_size(0),
        read_count(0),
        write_count(1),
        last_reads(0) {}

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

// Pool-wide statistics about replacements, shared by every attached process.
// One sample (the slot's read count at the moment it was replaced) is recorded
// per replace into a ring buffer, next to lifetime counters. All updates are
// relaxed atomics: snapshots may lag a few events, which is fine for metrics.
struct SharedStats {
  // Default ring capacity is two refresh cycles of the pool; callers that want
  // a wider window (and have budgeted the segment for it) pass it explicitly.
  static constexpr std::size_t kMinDefaultCapacity = 16;

  std::atomic<std::uint64_t> replace_events;
  std::atomic<std::uint64_t> reads_at_replace_total;
  std::atomic<std::uint64_t> reads_at_replace_max;
  std::atomic<std::uint64_t> blocked_replaces;
  std::atomic<std::uint64_t> cursor;
  std::size_t capacity;
  bi::offset_ptr<std::atomic<std::uint32_t>> samples;

  SharedStats(std::size_t cap, bi::offset_ptr<std::atomic<std::uint32_t>> buf)
      : replace_events(0),
        reads_at_replace_total(0),
        reads_at_replace_max(0),
        blocked_replaces(0),
        cursor(0),
        capacity(cap),
        samples(buf) {}

  SharedStats(const SharedStats&) = delete;
  SharedStats& operator=(const SharedStats&) = delete;

  void Record(std::uint64_t reads) {
    replace_events.fetch_add(1, std::memory_order_relaxed);
    reads_at_replace_total.fetch_add(reads, std::memory_order_relaxed);
    std::uint64_t seen = reads_at_replace_max.load(std::memory_order_relaxed);
    while (seen < reads && !reads_at_replace_max.compare_exchange_weak(
                               seen, reads, std::memory_order_relaxed)) {
    }
    const std::uint64_t slot = cursor.fetch_add(1, std::memory_order_relaxed);
    const auto clipped = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(reads, UINT32_MAX));
    samples[slot % capacity].store(clipped, std::memory_order_relaxed);
  }

  void Reset() {
    replace_events.store(0, std::memory_order_relaxed);
    reads_at_replace_total.store(0, std::memory_order_relaxed);
    reads_at_replace_max.store(0, std::memory_order_relaxed);
    blocked_replaces.store(0, std::memory_order_relaxed);
    cursor.store(0, std::memory_order_relaxed);
  }

  std::size_t SampleCount() const {
    return static_cast<std::size_t>(std::min<std::uint64_t>(
        cursor.load(std::memory_order_relaxed), capacity));
  }
};

// Plain copy of the pool-wide counters for consumption outside the segment.
struct StatsSnapshot {
  std::uint64_t replace_events;
  std::uint64_t reads_at_replace_total;
  std::uint64_t reads_at_replace_max;
  std::uint64_t blocked_replaces;
  std::size_t sample_count;
  std::size_t sample_capacity;
};

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
               std::optional<DType> dtype = std::nullopt, bool create = true,
               std::size_t stats_capacity = 0)
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
    stats_ = ResolveStats(stats_capacity, max_memory_size / chunk_size_);

    shared_resources_->logical_size->store(data_chunks_->size(),
                                           std::memory_order_relaxed);
  }

  ~SharedVector() { ScopedLock lock(shared_resources_->mutex); }

  SharedResources& GetSharedResources() { return *shared_resources_; }

  SharedChunksVector* GetDataChunks() const { return data_chunks_; }

  DType GetDtype() const noexcept { return dtype_; }

  // Pins the chunk for the lifetime of the returned pointer. `count_read`
  // records the access in the slot's read statistics; metadata lookups pass
  // false so one logical read is counted once.
  std::shared_ptr<SharedChunk> GetChunk(std::size_t index,
                                        bool count_read = true) {
    if (index >=
        shared_resources_->logical_size->load(std::memory_order_acquire)) {
      throw std::out_of_range("Index out of range");
    }

    SharedChunk* raw = (*data_chunks_)[index].get();

    {
      ScopedLock lock(raw->write_lock);
      raw->ref_count.fetch_add(1, std::memory_order_acq_rel);
    }
    if (count_read) {
      raw->read_count.fetch_add(1, std::memory_order_relaxed);
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
      if (stats_) {
        stats_->blocked_replaces.fetch_add(1, std::memory_order_relaxed);
      }
      throw std::runtime_error("Cannot replace chunk because it is in use");
    }

    const auto* begin = static_cast<const unsigned char*>(bytes);
    chunk->data->assign(begin, begin + nbytes);
    chunk->shape->assign(shape.begin(), shape.end());
    chunk->SetMeta(meta);

    const std::uint64_t reads =
        chunk->read_count.exchange(0, std::memory_order_relaxed);
    chunk->last_reads.store(reads, std::memory_order_relaxed);
    chunk->write_count.fetch_add(1, std::memory_order_relaxed);
    if (stats_) {
      stats_->Record(reads);
    }
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
    auto chunk = GetChunk(index, /*count_read=*/false);
    return std::vector<double>(chunk->meta, chunk->meta + chunk->meta_size);
  }

  // --- Read statistics -----------------------------------------------------

  std::uint64_t GetChunkReadCount(std::size_t index) const {
    return ChunkAt(index).read_count.load(std::memory_order_relaxed);
  }

  std::uint64_t GetChunkWriteCount(std::size_t index) const {
    return ChunkAt(index).write_count.load(std::memory_order_relaxed);
  }

  std::uint64_t GetChunkLastReads(std::size_t index) const {
    return ChunkAt(index).last_reads.load(std::memory_order_relaxed);
  }

  // Per-slot arrays over all `size()` slots, in slot order.
  std::vector<std::uint64_t> ReadCounts() const {
    return CollectChunkField(
        [](const SharedChunk& c) { return c.read_count.load(std::memory_order_relaxed); });
  }

  std::vector<std::uint64_t> WriteCounts() const {
    return CollectChunkField(
        [](const SharedChunk& c) { return c.write_count.load(std::memory_order_relaxed); });
  }

  std::vector<std::uint64_t> LastReads() const {
    return CollectChunkField(
        [](const SharedChunk& c) { return c.last_reads.load(std::memory_order_relaxed); });
  }

  bool HasStats() const noexcept { return stats_ != nullptr; }

  StatsSnapshot GetStatsSnapshot() const {
    const SharedStats& s = RequireStats();
    return StatsSnapshot{
        s.replace_events.load(std::memory_order_relaxed),
        s.reads_at_replace_total.load(std::memory_order_relaxed),
        s.reads_at_replace_max.load(std::memory_order_relaxed),
        s.blocked_replaces.load(std::memory_order_relaxed),
        s.SampleCount(),
        s.capacity};
  }

  // Recent reads-before-replacement samples, oldest first.
  std::vector<std::uint32_t> ReadSamples() const {
    const SharedStats& s = RequireStats();
    const std::uint64_t written = s.cursor.load(std::memory_order_relaxed);
    const std::size_t count = s.SampleCount();
    std::vector<std::uint32_t> out(count);
    const std::uint64_t start = written - count;  // oldest retained sample
    for (std::size_t i = 0; i < count; ++i) {
      out[i] = s.samples[(start + i) % s.capacity].load(std::memory_order_relaxed);
    }
    return out;
  }

  void ResetStats() { RequireStats().Reset(); }

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

  // Creates the statistics block when this process creates the segment;
  // attaching processes adopt the existing one (or none, for foreign segments).
  SharedStats* ResolveStats(std::size_t requested, std::size_t max_chunks) {
    auto& segment = shared_resources_->segment;
    SharedStats* existing = segment.find<SharedStats>("stats").first;
    if (existing) {
      return existing;
    }
    if (!data_chunks_->empty()) {
      return nullptr;  // populated segment without statistics support
    }
    const std::size_t capacity =
        requested > 0 ? requested
                      : std::max<std::size_t>(SharedStats::kMinDefaultCapacity,
                                              2 * max_chunks);
    auto* samples = segment.construct<std::atomic<std::uint32_t>>(
        bi::anonymous_instance)[capacity]();
    return segment.construct<SharedStats>("stats")(capacity, samples);
  }

  const SharedStats& RequireStats() const {
    if (!stats_) {
      throw std::runtime_error("Statistics are unavailable for this segment");
    }
    return *stats_;
  }

  SharedStats& RequireStats() {
    if (!stats_) {
      throw std::runtime_error("Statistics are unavailable for this segment");
    }
    return *stats_;
  }

  const SharedChunk& ChunkAt(std::size_t index) const {
    if (index >=
        shared_resources_->logical_size->load(std::memory_order_acquire)) {
      throw std::out_of_range("Index out of range");
    }
    const SharedChunkPtr& ptr = (*data_chunks_)[index];
    if (!ptr) {
      throw std::runtime_error("Chunk pointer is null");
    }
    return *ptr;
  }

  template <typename Fn>
  std::vector<std::uint64_t> CollectChunkField(Fn&& field) const {
    const std::size_t count =
        shared_resources_->logical_size->load(std::memory_order_acquire);
    std::vector<std::uint64_t> out(count);
    for (std::size_t i = 0; i < count; ++i) {
      out[i] = field(*(*data_chunks_)[i]);
    }
    return out;
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
  SharedStats* stats_ = nullptr;
  DType dtype_ = DType::kFloat32;
};

}  // namespace sharedvector
