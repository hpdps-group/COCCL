/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef MULTIBYTE_ANS_INCLUDE_UTILS_STATICUTILS_H
#define MULTIBYTE_ANS_INCLUDE_UTILS_STATICUTILS_H

#pragma once

#include <cuda.h>
#ifndef __host__
#define __host__
#define __device__
#endif

namespace multibyte_ans {

template <typename U, typename V>
constexpr __host__ __device__ auto divDown(U a, V b) -> decltype(a + b) {
  return (a / b);
}

template <typename U, typename V>
constexpr __host__ __device__ auto divUp(U a, V b) -> decltype(a + b) {
  return (a + b - 1) / b;
}

template <typename U, typename V>
constexpr __host__ __device__ auto roundDown(U a, V b) -> decltype(a + b) {
  return divDown(a, b) * b;
}

template <typename U, typename V>
constexpr __host__ __device__ auto roundUp(U a, V b) -> decltype(a + b) {
  return divUp(a, b) * b;
}

template <typename T>
constexpr __host__ __device__ bool isEvenDivisor(T a, T b) {
  return (a % b == 0) && ((a / b) >= 1);
}

template <class T>
constexpr __host__ __device__ T pow(T n, T power) {
  return (power > 0 ? n * pow(n, power - 1) : 1);
}

template <class T>
constexpr __host__ __device__ T pow2(T n) {
  return pow(2, (T)n);
}

template <typename T>
constexpr __host__ __device__ bool isPowerOf2(T v) {
  return (v && !(v & (v - 1)));
}

inline __host__ __device__ bool isPointerAligned(const void* p, int align) {
  return reinterpret_cast<uintptr_t>(p) % align == 0;
}

template <int Align>
inline __host__ __device__ uint32_t getAlignmentRoundUp(const void* p) {
  static_assert(isPowerOf2(Align), "");
  uint32_t diff = uint32_t(uintptr_t(p) & uintptr_t(Align - 1));
  return diff == 0 ? 0 : uint32_t(Align) - diff;
}

} 

#endif
