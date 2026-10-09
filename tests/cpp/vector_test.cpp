// Copyright 2024 Jonas Teuwen. All Rights Reserved.
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
#include "sharedvector/vector.h"

// C++ system headers
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

// Other libraries
#include "boost/interprocess/mapped_region.hpp"
#include "boost/interprocess/shared_memory_object.hpp"
#include "gtest/gtest.h"

// Project headers
#include "sharedvector/exceptions.h"

// Replace namespace using-directive with using-declarations
using sharedvector::DType;
using sharedvector::SharedChunk;
using sharedvector::SharedVector;
using sharedvector::StatsSnapshot;
using sharedvector::exceptions::MemoryError;
using sharedvector::exceptions::OutOfMemoryError;

// Fixture for SharedVector tests
class SharedVectorTest : public ::testing::Test {
 protected:
  const std::string shared_memory_name =
      "test_shared_memory_" + std::to_string(getpid());
  size_t chunk_size = 1024 * 1024;       // 1MB
  size_t overhead_per_chunk = 1 * 1024;  // 1KB overhead
  size_t max_memory_size = (10 * chunk_size) + (10 * overhead_per_chunk);
  SharedVector* shared_vector = nullptr;

  void SetUp() override {
    // Remove any previous shared memory segment before creating a new one
    boost::interprocess::shared_memory_object::remove(
        shared_memory_name.c_str());

    // Initialize a new SharedVector before each test
    shared_vector =
        new SharedVector(shared_memory_name, chunk_size, max_memory_size);
  }

  void TearDown() override {
    // Clean up shared memory and delete the vector after each test
    delete shared_vector;

    // Remove the shared memory segment after each test
    boost::interprocess::shared_memory_object::remove(
        shared_memory_name.c_str());
  }
};

// Test that a new SharedVector initializes correctly
TEST_F(SharedVectorTest, InitializationTest) {
  EXPECT_EQ(shared_vector->size(), 0);  // Should start with no chunks
  EXPECT_EQ(shared_vector->GetRefCount(),
            1);  // Only one reference (this instance)

  // Let's test multiple instances
  {
    SharedVector shared_vector2(shared_memory_name, chunk_size,
                                max_memory_size);
    EXPECT_EQ(shared_vector->GetRefCount(), 2);
  }
  // Now it's out of scope so the ref count should drop
  EXPECT_EQ(shared_vector->GetRefCount(), 1);

  // Let's also try to allocate it on the heap
  // Remember that this is not the preferred way to do it
  SharedVector* shared_vector2 =
      new SharedVector(shared_memory_name, chunk_size, max_memory_size);
  EXPECT_EQ(shared_vector->GetRefCount(), 2);
  delete shared_vector2;  // Properly delete the heap-allocated object
  EXPECT_EQ(shared_vector->GetRefCount(), 1);
}

// Test appending and retrieving data
TEST_F(SharedVectorTest, AppendAndGetTest) {
  std::vector<float> data = {1.0, 2.0, 3.0, 4.0};
  std::vector<size_t> shape = {2, 2};  // 2x2 array

  // Append data to shared vector
  shared_vector->append(data, shape);

  // Verify size after append
  EXPECT_EQ(shared_vector->size(), 1);

  // Get and verify the data
  auto chunk_ptr = shared_vector->GetChunk(0);
  EXPECT_EQ(chunk_ptr->Count<float>(), data.size());

  // Compare the data
  for (size_t i = 0; i < data.size(); i++) {
    EXPECT_EQ(chunk_ptr->As<float>()[i], data[i]);
  }

  // Compare the shape
  EXPECT_EQ(chunk_ptr->shape->size(), shape.size());
  for (size_t i = 0; i < shape.size(); i++) {
    EXPECT_EQ((*chunk_ptr->shape)[i], shape[i]);
  }
}

// Test replacing data
TEST_F(SharedVectorTest, ReplaceTest) {
  std::vector<float> data = {1.0, 2.0, 3.0, 4.0};
  std::vector<size_t> shape = {2, 2};  // 2x2 array
  shared_vector->append(data, shape);  // Add one item

  // Replace with new data
  std::vector<float> new_data = {5.0, 6.0, 7.0, 8.0};
  std::vector<size_t> new_shape = {2, 2};
  shared_vector->replace(0, new_data, new_shape);

  // Verify the replaced data
  auto chunk_ptr = shared_vector->GetChunk(0);
  EXPECT_EQ(chunk_ptr->Count<float>(), new_data.size());

  for (size_t i = 0; i < new_data.size(); i++) {
    EXPECT_EQ(chunk_ptr->As<float>()[i], new_data[i]);
  }

  // Verify the shape
  EXPECT_EQ(chunk_ptr->shape->size(), new_shape.size());
  for (size_t i = 0; i < new_shape.size(); i++) {
    EXPECT_EQ((*chunk_ptr->shape)[i], new_shape[i]);
  }
}

// Test replacing data with a smaller vector and different shape
TEST_F(SharedVectorTest, ReplaceWithSmallerVectorTest) {
  std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  std::vector<size_t> shape = {3, 2};  // Original shape: 3x2 array
  shared_vector->append(data, shape);  // Add initial data

  // Replace with a smaller vector and different shape
  std::vector<float> new_data = {7.0, 8.0, 9.0, 10.0};
  std::vector<size_t> new_shape = {2, 2};  // New shape: 2x2 array
  shared_vector->replace(0, new_data, new_shape);

  // Verify the replaced data size
  auto chunk_ptr = shared_vector->GetChunk(0);
  EXPECT_EQ(chunk_ptr->Count<float>(), new_data.size());

  // Verify the replaced data content
  for (size_t i = 0; i < new_data.size(); i++) {
    EXPECT_EQ(chunk_ptr->As<float>()[i], new_data[i]);
  }

  // Verify the new shape
  EXPECT_EQ(chunk_ptr->shape->size(), new_shape.size());
  for (size_t i = 0; i < new_shape.size(); i++) {
    EXPECT_EQ((*chunk_ptr->shape)[i], new_shape[i]);
  }
}

// Test replacing data with a larger vector and different shape
TEST_F(SharedVectorTest, ReplaceWithLargerVectorTest) {
  std::vector<float> data = {1.0, 2.0, 3.0, 4.0};
  std::vector<size_t> shape = {2, 2};  // Original shape: 2x2 array
  shared_vector->append(data, shape);  // Add initial data

  // Replace with a smaller vector and different shape
  std::vector<float> new_data = {7.0, 8.0, 9.0};
  std::vector<size_t> new_shape = {3, 1};  // New shape: 2x2 array
  shared_vector->replace(0, new_data, new_shape);

  // Verify the replaced data size
  auto chunk_ptr = shared_vector->GetChunk(0);
  EXPECT_EQ(chunk_ptr->Count<float>(), new_data.size());

  // Verify the replaced data content
  for (size_t i = 0; i < new_data.size(); i++) {
    EXPECT_EQ(chunk_ptr->As<float>()[i], new_data[i]);
  }

  // Verify the new shape
  EXPECT_EQ(chunk_ptr->shape->size(), new_shape.size());
  for (size_t i = 0; i < new_shape.size(); i++) {
    EXPECT_EQ((*chunk_ptr->shape)[i], new_shape[i]);
  }
}

// Test reference count management
TEST_F(SharedVectorTest, RefCountTest) {
  std::vector<float> data = {1.0, 2.0, 3.0, 4.0};
  std::vector<size_t> shape = {2, 2};
  shared_vector->append(data, shape);

  // Get data and check reference count
  // This is encapsulated in { } to ensure chunk_ptr goes out of scope
  {
    auto chunk_ptr = shared_vector->GetChunk(0);
    EXPECT_EQ(shared_vector->GetChunkRefCount(0), 1);
  }
  // After chunk goes out of scope, ref count should decrease
  EXPECT_EQ(shared_vector->GetChunkRefCount(0), 0);
}

// Test out of range access
TEST_F(SharedVectorTest, OutOfRangeTest) {
  EXPECT_THROW(shared_vector->GetChunk(0),
               std::out_of_range);  // No data, index 0 should fail
}

TEST_F(SharedVectorTest, AppendOversizedArrayTest) {
  // Create a data array larger than the chunk size
  // Assuming chunk_size is 1 MB, we'll create an array slightly larger

  // Calculate the number of elements to exceed chunk size
  size_t elements_to_exceed_chunk = (chunk_size / sizeof(float)) + 1;
  std::vector<float> oversized_data(elements_to_exceed_chunk, 1.0f);
  std::vector<size_t> shape = {elements_to_exceed_chunk};

  // Attempt to append the oversized array and expect an exception
  EXPECT_THROW(shared_vector->append(oversized_data, shape), MemoryError);

  // Let's also try to append a smaller array which is precisely correct
  std::vector<float> correct_data(chunk_size / sizeof(float), 1.0f);
  shared_vector->append(correct_data, shape);
  EXPECT_EQ(shared_vector->size(), 1);
}

// Large number of samples test
TEST_F(SharedVectorTest, LargeNumberOfSmallSamplesTest) {
  bool out_of_memory = false;
  std::size_t chunk_count = 0;
  while (!out_of_memory) {
    std::vector<float> data = {static_cast<float>(chunk_count)};
    std::vector<size_t> shape = {1};
    try {
      shared_vector->append(data, shape);
      chunk_count++;
    } catch (const OutOfMemoryError&) {
      out_of_memory = true;
    }
  }
  EXPECT_EQ(shared_vector->size(), chunk_count);
  for (size_t i = 0; i < chunk_count; ++i) {
    auto chunk_ptr = shared_vector->GetChunk(i);
    EXPECT_EQ(chunk_ptr->As<float>()[0], static_cast<float>(i));
  }
}

// Concurrency test
TEST_F(SharedVectorTest, SequentialWriterReaderTest) {
  // Writer thread
  std::thread writer([&]() {
    try {
      for (size_t i = 0; i < 10; ++i) {
        // Ensure array fits within chunk size (1MB)
        std::vector<float> data(128 * 128, static_cast<float>(i));
        std::vector<size_t> shape = {128 * 128};

        shared_vector->append(data, shape);

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    } catch (const std::exception& e) {
      std::cerr << "Exception in writer thread: " << e.what() << std::endl;
    }
  });

  // Wait for the writer to finish
  writer.join();

  ASSERT_EQ(shared_vector->size(), 10);

  // Reader thread
  std::thread reader([&]() {
    for (size_t i = 0; i < 10; ++i) {
      auto chunk_ptr = shared_vector->GetChunk(i);
      EXPECT_EQ(chunk_ptr->As<float>()[0], static_cast<float>(i));
    }
  });

  // Wait for the reader to finish
  reader.join();

  // Final check
  EXPECT_EQ(shared_vector->size(), 10);
}

TEST_F(SharedVectorTest, ConcurrentWriterReaderTest) {
  // Shared variable to signal completion
  std::atomic<bool> writer_done(false);

  // Writer thread
  std::thread writer([&]() {
    try {
      for (size_t i = 0; i < 10; ++i) {
        std::vector<float> data(128 * 128, static_cast<float>(i));
        std::vector<size_t> shape = {128 * 128};

        shared_vector->append(data, shape);

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      writer_done = true;
    } catch (const std::exception& e) {
      std::cerr << "Exception in writer thread: " << e.what() << std::endl;
    }
  });

  // Reader thread
  std::thread reader([&]() {
    size_t read_index = 0;
    while (!writer_done || read_index < shared_vector->size()) {
      if (read_index < shared_vector->size()) {
        auto chunk_ptr = shared_vector->GetChunk(read_index);
        EXPECT_EQ(chunk_ptr->As<float>()[0], static_cast<float>(read_index));
        read_index++;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  });

  // Wait for both threads to finish
  writer.join();
  reader.join();

  // Final check
  EXPECT_EQ(shared_vector->size(), 10);
}

TEST_F(SharedVectorTest, ConcurrentAppendTest) {
  const size_t num_threads = 4;
  const size_t appends_per_thread = 2;
  std::vector<std::thread> writer_threads;
  std::mutex output_mutex;  // For synchronizing console output (optional)

  // Vector to keep track of data each thread appends
  std::vector<std::vector<float>> expected_data;
  expected_data.resize(num_threads);

  // Seed the random number generator
  unsigned int seed = 314;

  // Start writer threads
  for (size_t t = 0; t < num_threads; ++t) {
    writer_threads.emplace_back([&, t]() {
      for (size_t i = 0; i < appends_per_thread; ++i) {
        // Generate unique data for each append
        float value = static_cast<float>(t * 1000 + i);
        std::vector<float> data = {value};
        std::vector<size_t> shape = {1};

        // Simulate random work with thread-safe rand_r
        unsigned int local_seed = seed + t;
        std::this_thread::sleep_for(
            std::chrono::milliseconds(rand_r(&local_seed) % 10));

        try {
          shared_vector->append(data, shape);

          // Record the data appended by this thread
          expected_data[t].push_back(value);
        } catch (const std::exception& e) {
          std::lock_guard<std::mutex> lock(output_mutex);
          std::cerr << "Exception in writer thread " << t << ": " << e.what()
                    << std::endl;
        }
      }
    });
  }

  // Wait for all writer threads to finish
  for (auto& thread : writer_threads) {
    thread.join();
  }

  // Verify that all data has been appended
  size_t total_expected_appends = num_threads * appends_per_thread;
  EXPECT_EQ(shared_vector->size(), total_expected_appends);

  // Collect the data from shared_vector
  std::vector<float> actual_data;
  for (size_t i = 0; i < shared_vector->size(); ++i) {
    auto chunk_ptr = shared_vector->GetChunk(i);
    actual_data.push_back(chunk_ptr->As<float>()[0]);
  }

  // Verify that all expected data is present in actual_data
  // Need to sort before comparison as order is arbitrary
  std::vector<float> all_expected_values;
  for (const auto& thread_data : expected_data) {
    all_expected_values.insert(all_expected_values.end(), thread_data.begin(),
                               thread_data.end());
  }

  std::sort(all_expected_values.begin(), all_expected_values.end());
  std::sort(actual_data.begin(), actual_data.end());

  EXPECT_EQ(all_expected_values, actual_data);
}

// Test to detect extra memory usage per chunk
TEST_F(SharedVectorTest, MemoryUsageTest) {
  size_t max_acceptable_overhead_per_chunk = 512;
  // Remove any previous shared memory segment before creating a new one
  boost::interprocess::shared_memory_object::remove(shared_memory_name.c_str());

  // Initialize a SharedVector with specified parameters
  SharedVector test_vector(shared_memory_name, chunk_size, max_memory_size);

  // Prepare data that fills up one chunk
  size_t data_elements = chunk_size / sizeof(float);
  std::vector<float> data(data_elements, 1.0f);
  std::vector<size_t> shape = {data_elements};

  // Calculate expected number of chunks we can store
  size_t expected_chunks =
      max_memory_size / (chunk_size + max_acceptable_overhead_per_chunk);

  size_t actual_chunks = 0;
  bool out_of_memory = false;
  while (!out_of_memory) {
    try {
      std::size_t free_memory_before = test_vector.GetFreeMemory();
      test_vector.append(data, shape);
      std::size_t free_memory_after = test_vector.GetFreeMemory();
      std::size_t used_memory = free_memory_before - free_memory_after;

      // Verify that used memory is within acceptable bounds
      ASSERT_LE(used_memory, chunk_size + max_acceptable_overhead_per_chunk);

      actual_chunks++;
    } catch (const OutOfMemoryError&) {
      out_of_memory = true;
    }
  }

  // Check that the actual number of chunks matches the expected number
  EXPECT_EQ(actual_chunks, expected_chunks);

  // Clean up
  boost::interprocess::shared_memory_object::remove(shared_memory_name.c_str());
}

// Test to demonstrate pointer invalidation after resize
TEST_F(SharedVectorTest, PointerInvalidationTest) {
  // #1: Append initial chunks and cache their pointers
  std::vector<std::shared_ptr<SharedChunk>> cached_chunks;
  size_t initial_append_count = 2;

  for (size_t i = 0; i < initial_append_count; ++i) {
    std::vector<float> data = {static_cast<float>(i)};
    std::vector<size_t> shape = {1};

    shared_vector->append(data, shape);
    auto chunk_ptr = shared_vector->GetChunk(i);
    cached_chunks.push_back(chunk_ptr);
  }

  // #2: Append additional chunks to trigger resize
  size_t additional_append_count = 4;  // Total chunks now 6
  for (size_t i = initial_append_count;
       i < initial_append_count + additional_append_count; ++i) {
    std::vector<float> data = {static_cast<float>(i)};
    std::vector<size_t> shape = {1};

    shared_vector->append(data, shape);
  }

  // #3: Access cached pointers after resize
  for (size_t i = 0; i < initial_append_count; ++i) {
    auto cached_chunk_ptr = cached_chunks[i];
    try {
      // Attempt to access the data
      float value = cached_chunk_ptr->As<float>()[0];
      // Compare with expected value
      EXPECT_EQ(value, static_cast<float>(i))
          << "Data mismatch for chunk " << i;
    } catch (const std::exception& e) {
      // If accessing data throws an exception, it indicates corruption
      FAIL() << "Exception accessing cached chunk " << i << ": " << e.what();
    }
  }

  // #4: Additionally, verify that the chunks in the vector have correct data
  for (size_t i = 0; i < initial_append_count + additional_append_count; ++i) {
    auto chunk_ptr = shared_vector->GetChunk(i);
    EXPECT_EQ(chunk_ptr->As<float>()[0], static_cast<float>(i))
        << "Data mismatch for chunk " << i;
  }
}

TEST_F(SharedVectorTest, SharedVectorLogicalSizeChanges) {
  // Arrange
  std::vector<float> data = {1.0, 2.0, 3.0, 4.0};
  std::vector<size_t> shape = {2, 2};

  // Act
  auto size_pre_append = shared_vector->size();
  shared_vector->append(data, shape);
  auto size_after_append = shared_vector->size();

  // Assert
  EXPECT_EQ(size_pre_append, 0);
  EXPECT_EQ(size_after_append, 1);
}

TEST_F(SharedVectorTest, SharedVectorLogicalSizeChangesForSecondPointer) {
  // Arrange
  std::vector<float> data = {1.0, 2.0, 3.0, 4.0};
  std::vector<size_t> shape = {2, 2};
  SharedVector shared_vector_second_instance(shared_memory_name, chunk_size,
                                             max_memory_size);

  // Act
  auto size_pre_append = shared_vector_second_instance.size();
  // Vector initialized at setup appends, updates logical size globally
  shared_vector->append(data, shape);
  auto size_after_append = shared_vector_second_instance.size();

  // Assert
  EXPECT_EQ(shared_vector->GetRefCount(), 2);
  EXPECT_EQ(size_pre_append, 0);
  EXPECT_EQ(size_after_append, 1);
}

TEST_F(SharedVectorTest, CorrectLogicalSizeOnInstantiation) {
  // Arrange
  std::vector<float> data = {1.0, 2.0, 3.0, 4.0};
  std::vector<size_t> shape = {2, 2};

  // Act
  shared_vector->append(data, shape);
  SharedVector shared_vector_second_instance(shared_memory_name, chunk_size,
                                             max_memory_size);

  // Assert
  EXPECT_EQ(shared_vector_second_instance.size(), 1);
}

// Regression test: a reader acquiring a chunk must never observe a torn write
// from a concurrent replace(). Each written generation has every element equal
// to its value and a shape whose product equals the data size, so a torn read
// shows up as a value/shape/size mismatch.
TEST_F(SharedVectorTest, ConcurrentGetAndReplaceNeverTearsRead) {
  // Arrange
  struct Generation {
    float value;
    std::vector<float> data;
    std::vector<size_t> shape;
  };
  const Generation gen_a{1.0F, std::vector<float>(4, 1.0F), {2, 2}};
  const Generation gen_b{2.0F, std::vector<float>(9, 2.0F), {3, 3}};
  shared_vector->append(gen_a.data, gen_a.shape);

  constexpr int kIterations = 50000;
  std::atomic<bool> torn{false};
  std::atomic<bool> done{false};

  // Act
  std::thread writer([&] {
    for (int i = 0; i < kIterations; ++i) {
      const Generation& gen = (i % 2 == 0) ? gen_b : gen_a;
      try {
        shared_vector->replace(0, gen.data, gen.shape);
      } catch (const std::runtime_error&) {
        // Expected: chunk is pinned by the reader, skip this round.
      }
    }
    done.store(true, std::memory_order_release);
  });

  std::thread reader([&] {
    while (!done.load(std::memory_order_acquire)) {
      auto chunk = shared_vector->GetChunk(0);
      const float* data = chunk->As<float>();
      const size_t count = chunk->Count<float>();
      const auto& shape = *chunk->shape;
      size_t product = 1;
      for (size_t dim : shape) product *= dim;
      const bool value_ok = data[0] == gen_a.value || data[0] == gen_b.value;
      const bool consistent =
          product == count &&
          std::all_of(data, data + count,
                      [&](float v) { return v == data[0]; });
      if (!value_ok || !consistent) {
        torn.store(true, std::memory_order_release);
        return;
      }
    }
  });

  writer.join();
  reader.join();

  // Assert
  EXPECT_FALSE(torn.load(std::memory_order_acquire));
}

TEST_F(SharedVectorTest, Int16AppendAndGetBytes) {
  // Arrange
  const std::string name = "test_shared_memory_int16";
  boost::interprocess::shared_memory_object::remove(name.c_str());
  SharedVector vector(name, chunk_size, max_memory_size, DType::kInt16);
  std::vector<int16_t> data = {1, -2, 3};

  // Act
  vector.append(data, {3});
  auto chunk = vector.GetChunk(0);

  // Assert
  EXPECT_EQ(vector.GetDtype(), DType::kInt16);
  EXPECT_EQ(chunk->data->size(), 6U);
  EXPECT_EQ(chunk->Count<int16_t>(), 3U);
  EXPECT_EQ(chunk->As<int16_t>()[1], -2);
  chunk.reset();
  boost::interprocess::shared_memory_object::remove(name.c_str());
}

TEST_F(SharedVectorTest, AppendWrongTypedVectorThrows) {
  // Arrange
  std::vector<int16_t> data = {1, 2};

  // Act & Assert
  EXPECT_THROW(shared_vector->append(data, {2}), std::invalid_argument);
  EXPECT_EQ(shared_vector->size(), 0U);
}

TEST_F(SharedVectorTest, AttachWithConflictingDtypeThrows) {
  // Act & Assert
  EXPECT_THROW(SharedVector(shared_memory_name, chunk_size, max_memory_size,
                            DType::kInt16),
               std::invalid_argument);
}

TEST_F(SharedVectorTest, AttachWithoutDtypeAdopts) {
  // Arrange
  const std::string name = "test_shared_memory_adopt";
  boost::interprocess::shared_memory_object::remove(name.c_str());
  SharedVector producer(name, chunk_size, max_memory_size, DType::kInt16);

  // Act
  SharedVector consumer(name, chunk_size, max_memory_size);

  // Assert
  EXPECT_EQ(consumer.GetDtype(), DType::kInt16);
  boost::interprocess::shared_memory_object::remove(name.c_str());
}

TEST_F(SharedVectorTest, MetaRoundTrip) {
  // Arrange
  std::vector<float> data = {1.0F, 2.0F};
  std::vector<double> meta = {0.5, 0.7, 2.0};

  // Act
  shared_vector->append(data, {2}, meta);
  std::vector<double> stored = shared_vector->GetChunkMeta(0);
  shared_vector->replace(0, data, {2});
  std::vector<double> cleared = shared_vector->GetChunkMeta(0);

  // Assert
  EXPECT_EQ(stored, meta);
  EXPECT_TRUE(cleared.empty());
}

TEST_F(SharedVectorTest, MetaOverCapacityThrows) {
  // Arrange
  std::vector<float> data = {1.0F};
  std::vector<double> meta(SharedChunk::kMetaCapacity + 1, 1.0);

  // Act & Assert
  EXPECT_THROW(shared_vector->append(data, {1}, meta), std::invalid_argument);
  EXPECT_EQ(shared_vector->size(), 0U);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

// Attaching with create=false must fail when the segment does not exist.
TEST_F(SharedVectorTest, OpenOnlyMissingSegmentThrows) {
  const std::string missing = shared_memory_name + "_missing";
  boost::interprocess::shared_memory_object::remove(missing.c_str());
  EXPECT_THROW(SharedVector(missing, chunk_size, max_memory_size, std::nullopt,
                            /*create=*/false),
               boost::interprocess::interprocess_exception);
  EXPECT_FALSE(boost::interprocess::shared_memory_object::remove(missing.c_str()));
}

// Attaching with create=false to an existing segment sees the producer's data.
TEST_F(SharedVectorTest, OpenOnlyAttachesToExistingSegment) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f};
  shared_vector->append(data, {3});

  SharedVector consumer(shared_memory_name, 4096, 65536, std::nullopt,
                        /*create=*/false);

  EXPECT_EQ(consumer.size(), 1);
  EXPECT_EQ(consumer.GetRefCount(), 2);
  auto chunk = consumer.GetChunk(0);
  EXPECT_EQ(chunk->Count<float>(), 3);
  EXPECT_FLOAT_EQ(chunk->As<float>()[2], 3.0f);
}

// --- Read statistics ---------------------------------------------------------

TEST_F(SharedVectorTest, ReadCountIncrementsOnGetChunkAndNotOnMeta) {
  shared_vector->append(std::vector<float>{1.0f, 2.0f}, {2}, {0.5});

  {
    auto a = shared_vector->GetChunk(0);
    auto b = shared_vector->GetChunk(0);
    EXPECT_EQ(shared_vector->GetChunkReadCount(0), 2);
  }
  shared_vector->GetChunkMeta(0);
  shared_vector->GetChunkShape(0);
  shared_vector->GetChunkRefCount(0);

  EXPECT_EQ(shared_vector->GetChunkReadCount(0), 2);
  EXPECT_EQ(shared_vector->GetChunkWriteCount(0), 1);
  EXPECT_EQ(shared_vector->GetChunkLastReads(0), 0);
}

TEST_F(SharedVectorTest, ReplaceCapturesReadsAndResetsCounter) {
  shared_vector->append(std::vector<float>{1.0f}, {1});
  for (int i = 0; i < 3; ++i) {
    shared_vector->GetChunk(0);
  }

  shared_vector->replace(0, std::vector<float>{2.0f}, {1});

  EXPECT_EQ(shared_vector->GetChunkReadCount(0), 0);
  EXPECT_EQ(shared_vector->GetChunkLastReads(0), 3);
  EXPECT_EQ(shared_vector->GetChunkWriteCount(0), 2);
  StatsSnapshot s = shared_vector->GetStatsSnapshot();
  EXPECT_EQ(s.replace_events, 1);
  EXPECT_EQ(s.reads_at_replace_total, 3);
  EXPECT_EQ(s.reads_at_replace_max, 3);
  EXPECT_EQ(s.blocked_replaces, 0);
  EXPECT_EQ(s.sample_count, 1);
  EXPECT_EQ(shared_vector->ReadSamples(), (std::vector<std::uint32_t>{3}));
}

TEST_F(SharedVectorTest, BlockedReplaceIsCounted) {
  shared_vector->append(std::vector<float>{1.0f}, {1});
  auto pinned = shared_vector->GetChunk(0);

  EXPECT_THROW(shared_vector->replace(0, std::vector<float>{2.0f}, {1}),
               std::runtime_error);

  EXPECT_EQ(shared_vector->GetStatsSnapshot().blocked_replaces, 1);
  EXPECT_EQ(shared_vector->GetStatsSnapshot().replace_events, 0);
  EXPECT_EQ(shared_vector->GetChunkReadCount(0), 1);  // pinned read still counts
}

TEST_F(SharedVectorTest, StatsTrackTotalsAndMaxAcrossSlots) {
  shared_vector->append(std::vector<float>{1.0f}, {1});
  shared_vector->append(std::vector<float>{2.0f}, {1});
  for (int i = 0; i < 5; ++i) shared_vector->GetChunk(0);
  shared_vector->GetChunk(1);

  shared_vector->replace(0, std::vector<float>{3.0f}, {1});
  shared_vector->replace(1, std::vector<float>{4.0f}, {1});
  shared_vector->replace(1, std::vector<float>{5.0f}, {1});

  StatsSnapshot s = shared_vector->GetStatsSnapshot();
  EXPECT_EQ(s.replace_events, 3);
  EXPECT_EQ(s.reads_at_replace_total, 6);
  EXPECT_EQ(s.reads_at_replace_max, 5);
  EXPECT_EQ(shared_vector->ReadSamples(), (std::vector<std::uint32_t>{5, 1, 0}));
  EXPECT_EQ(shared_vector->LastReads(), (std::vector<std::uint64_t>{5, 0}));
  EXPECT_EQ(shared_vector->WriteCounts(), (std::vector<std::uint64_t>{2, 3}));
  EXPECT_EQ(shared_vector->ReadCounts(), (std::vector<std::uint64_t>{0, 0}));
}

TEST_F(SharedVectorTest, RingBufferWrapsKeepingNewestInOrder) {
  const std::string name = shared_memory_name + "_ring";
  boost::interprocess::shared_memory_object::remove(name.c_str());
  {
    SharedVector vec(name, chunk_size, max_memory_size, std::nullopt, true,
                     /*stats_capacity=*/4);
    vec.append(std::vector<float>{1.0f}, {1});
    for (int round = 1; round <= 6; ++round) {
      for (int i = 0; i < round; ++i) vec.GetChunk(0);
      vec.replace(0, std::vector<float>{1.0f}, {1});
    }

    StatsSnapshot s = vec.GetStatsSnapshot();
    EXPECT_EQ(s.sample_capacity, 4);
    EXPECT_EQ(s.sample_count, 4);
    EXPECT_EQ(s.replace_events, 6);
    EXPECT_EQ(vec.ReadSamples(), (std::vector<std::uint32_t>{3, 4, 5, 6}));
  }
  boost::interprocess::shared_memory_object::remove(name.c_str());
}

TEST_F(SharedVectorTest, ResetStatsZeroesPoolCountersOnly) {
  shared_vector->append(std::vector<float>{1.0f}, {1});
  shared_vector->GetChunk(0);
  shared_vector->replace(0, std::vector<float>{2.0f}, {1});

  shared_vector->ResetStats();

  StatsSnapshot s = shared_vector->GetStatsSnapshot();
  EXPECT_EQ(s.replace_events, 0);
  EXPECT_EQ(s.reads_at_replace_max, 0);
  EXPECT_EQ(s.sample_count, 0);
  EXPECT_TRUE(shared_vector->ReadSamples().empty());
  EXPECT_EQ(shared_vector->GetChunkLastReads(0), 1);
  EXPECT_EQ(shared_vector->GetChunkWriteCount(0), 2);
}

TEST_F(SharedVectorTest, AttachingProcessSharesStatsAndDefaultCapacity) {
  shared_vector->append(std::vector<float>{1.0f}, {1});
  SharedVector consumer(shared_memory_name, 4096, 65536, std::nullopt,
                        /*create=*/false);

  consumer.GetChunk(0);
  consumer.GetChunk(0);
  shared_vector->replace(0, std::vector<float>{2.0f}, {1});

  EXPECT_TRUE(consumer.HasStats());
  EXPECT_EQ(consumer.GetStatsSnapshot().replace_events, 1);
  EXPECT_EQ(consumer.ReadSamples(), (std::vector<std::uint32_t>{2}));
  // default capacity = 2 * max_chunks, with max_chunks = 10 for this fixture
  EXPECT_EQ(consumer.GetStatsSnapshot().sample_capacity, 20);
}
