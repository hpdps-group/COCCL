cmake_minimum_required(VERSION 3.18)
project(coccl_dietgpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(WITH_GFLAGS OFF CACHE BOOL "" FORCE)
set(WITH_GTEST OFF CACHE BOOL "" FORCE)
set(WITH_GMOCK OFF CACHE BOOL "" FORCE)
set(WITH_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(WITH_UNWIND OFF CACHE BOOL "" FORCE)
add_subdirectory(vendor/glog EXCLUDE_FROM_ALL)

add_library(dietgpu SHARED
  dietgpu_plugin.cu
  vendor/dietgpu/dietgpu/ans/GpuANSDecode.cu
  vendor/dietgpu/dietgpu/ans/GpuANSEncode.cu
  vendor/dietgpu/dietgpu/ans/GpuANSInfo.cu
  vendor/dietgpu/dietgpu/utils/DeviceUtils.cpp
  vendor/dietgpu/dietgpu/utils/StackDeviceMemory.cpp)

target_include_directories(dietgpu PRIVATE
  ${COCCL_INCLUDE_DIR}
  ${COCCL_NCCL_INCLUDE_DIR}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}/vendor/dietgpu)
target_link_libraries(dietgpu PRIVATE glog::glog)
target_compile_options(dietgpu PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
set_target_properties(dietgpu PROPERTIES
  CUDA_ARCHITECTURES "${COCCL_CUDA_ARCHITECTURES}"
  CUDA_SEPARABLE_COMPILATION OFF
  OUTPUT_NAME dietgpu)
