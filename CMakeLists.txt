cmake_minimum_required(VERSION 3.24)  # required by cmake/conan_provider.cmake

option(SHAREDVECTOR_USE_CONAN "Resolve Boost/GTest through the vendored cmake-conan provider" ON)
option(SHAREDVECTOR_BUILD_PYTHON "Build the pybind11 extension module" ON)
option(BUILD_TESTING "Build the GoogleTest suite (pulls GTest through Conan)" OFF)

# The dependency provider is picked up by project(), so everything it reads has to be set first.
if(SHAREDVECTOR_USE_CONAN)
  if(NOT CMAKE_PROJECT_TOP_LEVEL_INCLUDES)
    set(CMAKE_PROJECT_TOP_LEVEL_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/conan_provider.cmake")
  endif()
  set(_sharedvector_conan_args "--build=missing")
  if(BUILD_TESTING)
    list(APPEND _sharedvector_conan_args "-o" "&:with_tests=True")
  endif()
  set(CONAN_INSTALL_ARGS "${_sharedvector_conan_args}" CACHE STRING "Arguments appended to conan install")
endif()

project(sharedvector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)  # read by the provider -> compiler.cppstd=17
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# PEP 517 build frontends usually put the isolated environment's bin/ on PATH; when one does not,
# fall back to the scripts directory of the interpreter scikit-build-core hands us.
if(SHAREDVECTOR_USE_CONAN AND NOT CONAN_COMMAND)
  find_program(CONAN_COMMAND conan)
  if(NOT CONAN_COMMAND AND Python_EXECUTABLE)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('scripts'))"
      OUTPUT_VARIABLE _sharedvector_scripts_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    find_program(CONAN_COMMAND conan HINTS "${_sharedvector_scripts_dir}" REQUIRED)
  endif()
endif()

# The first find_package() triggers `conan install`.
find_package(Boost 1.87 REQUIRED)
find_package(Threads REQUIRED)

add_library(sharedvector_core INTERFACE)
target_include_directories(sharedvector_core INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/include")
target_link_libraries(sharedvector_core INTERFACE Boost::headers Threads::Threads)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(sharedvector_core INTERFACE rt)  # shm_open lives in librt on glibc < 2.34
endif()

if(SHAREDVECTOR_BUILD_PYTHON)
  find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_vector MODULE src/cpp/python/module.cpp)
  target_link_libraries(_vector PRIVATE sharedvector_core)
  install(TARGETS _vector LIBRARY DESTINATION sharedvector)
endif()

if(BUILD_TESTING)
  enable_testing()
  find_package(GTest REQUIRED)
  add_executable(vector_test tests/cpp/vector_test.cpp)
  target_link_libraries(vector_test PRIVATE sharedvector_core GTest::gtest GTest::gtest_main)
  add_test(NAME vector_test COMMAND vector_test)
endif()
