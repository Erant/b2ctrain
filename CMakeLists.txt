cmake_minimum_required(VERSION 3.24)
project(b2ctrain LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES)
  # Ada (RTX 40 / L40S) and consumer Blackwell (RTX 50), plus PTX for newer parts.
  set(CMAKE_CUDA_ARCHITECTURES "89-real;120-real;120-virtual")
endif()
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Threads REQUIRED)

# The CUDA headers for the HOST compiler. nvcc finds cuda_fp16.h and friends
# on its own, but src/gpu/*.cuh is included from plain .cpp translation
# units too (trainer.cpp through gpu/render.h, and every other consumer of
# a device struct), and c++ has no idea where the toolkit lives. On a box
# with a distro CUDA package this works by accident — Ubuntu drops a copy
# in /usr/include — and then fails in a container built on
# nvidia/cuda:*-devel, where the toolkit is only under /usr/local/cuda.
# Caught by b2crunner's image build (2026-09-07).
find_package(CUDAToolkit REQUIRED)

add_library(b2c_core STATIC
  src/util/log.cpp
  src/cli.cpp
  src/dataset/colmap.cpp
  src/dataset/views.cpp
  src/dataset/mesh.cpp
  src/ply.cpp
  src/model.cu
  src/gpu/render.cu
  src/gpu/project.cu
  src/gpu/binning.cu
  src/gpu/raster_fwd.cu
  src/gpu/raster_bwd.cu
  src/gpu/raster_bwd_tc.cu
  src/gpu/loss.cu
  src/gpu/optim.cu
  src/gpu/deform.cu
  src/gpu/refine.cu
  src/gpu/images.cu
  src/gpu/align.cu
  src/gpu/meshdepth.cu
  src/gpu/probe.cu
  src/train/trainer.cpp
  src/train/init.cpp
  src/train/evidence.cu
  src/render/renderer.cpp
  src/render/confidence.cu
  src/render/probe.cpp
)
target_include_directories(b2c_core PUBLIC src third_party ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(b2c_core PUBLIC Threads::Threads)
target_compile_options(b2c_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Wno-unknown-pragmas -O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo --expt-relaxed-constexpr -Xcudafe=--diag_suppress=20012 -Wno-deprecated-gpu-targets>
)
set_target_properties(b2c_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

add_executable(b2ctrain src/main.cpp)
target_link_libraries(b2ctrain PRIVATE b2c_core)

enable_testing()
add_executable(b2c_tests tests/tests.cpp tests/reference.cpp)
target_link_libraries(b2c_tests PRIVATE b2c_core)
add_test(NAME b2c_tests COMMAND b2c_tests)
