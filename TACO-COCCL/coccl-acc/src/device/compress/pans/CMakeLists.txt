# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.
#set(CMAKE_MODULE_PATH "/usr/local/python3/lib/python3.8/site-packages/torch/cmake/modules" ${CMAKE_MODULE_PATH})
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
set(CMAKE_CUDA_COMPILER ${CUDACXX})
set(CMAKE_CXX_COMPILER ${CXX})
project(multibyte_ans LANGUAGES CUDA CXX VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
include(CheckLanguage)
check_language(CUDA)

#find_package(Torch REQUIRED)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_package(CUDA REQUIRED)

if(${CMAKE_VERSION} VERSION_LESS_EQUAL "3.13.4")
    cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto")
    message("ARCH_FLAGS = ${ARCH_FLAGS}")
    string(REPLACE "-gencode;" "--generate-code=" ARCH_FLAGS "${ARCH_FLAGS}")
    string(APPEND CMAKE_CUDA_FLAGS "${ARCH_FLAGS}")
else()
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
    set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
endif()

# Set default build type.
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
    FORCE
  )
endif()
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_library(gpu_ans SHARED
  #src/GpuANSDecode.cu
  src/GpuANS.cu
)

target_include_directories(gpu_ans PUBLIC
  "${CUDA_INCLUDE_DIRS}"
  "${TORCH_INCLUDE_DIRS}"
)
target_link_libraries(gpu_ans PUBLIC
  ${CUDA_LIBRARIES}
  "${TORCH_LIBRARIES}"
)

target_include_directories(gpu_ans
      PUBLIC
      # where the library itself will look for its internal headers
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      PUBLIC
      # where top-level project will look for the library's public headers
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      # where external projects will look for the library's public headers
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

add_library(pans SHARED coccl_interface/pans.cu)
# set_source_files_properties(coccl_interface/pans.c PROPERTIES LANGUAGE CXX)


target_link_libraries(pans PRIVATE gpu_ans)

set(COCCL_INCLUDE_DIRS ../../../include CACHE STRING "COCCL_INCLUDE_DIRS")
set(NCCL_INCLUDE_DIRS ../../../../build/include CACHE STRING "NCCL_INCLUDE_DIRS")

message(STATUS "COCCL_INCLUDE_DIRS = ${COCCL_INCLUDE_DIRS}")


target_include_directories(pans PRIVATE ${CUDA_INCLUDE_DIRS} ${COCCL_INCLUDE_DIRS} ${NCCL_INCLUDE_DIRS})
target_link_libraries(pans PRIVATE ${CUDA_CUDART_LIBRARY} stdc++)


install(TARGETS pans EXPORT pans-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})


message(LIBRARY_OUTPUT_DIRECTORY "LIBRARY_OUTPUT_DIRECTORY: ${LIBRARY_OUTPUT_DIRECTORY}")
