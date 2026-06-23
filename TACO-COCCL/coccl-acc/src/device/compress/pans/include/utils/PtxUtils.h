/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef MULTIBYTE_ANS_INCLUDE_UTILS_PTXUTILS_H
#define MULTIBYTE_ANS_INCLUDE_UTILS_PTXUTILS_H

#pragma once

#include <cuda.h>

namespace multibyte_ans {

__device__ __forceinline__ int getLaneId() {
  int laneId;
  asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
  return laneId;
}

__device__ __forceinline__ unsigned getLaneMaskLt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__ unsigned getLaneMaskLe() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__ unsigned getLaneMaskGt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__ unsigned getLaneMaskGe() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_ge;" : "=r"(mask));
  return mask;
}

template <typename T, int Width = kWarpSize>
__device__ inline T warpReduceAllSum(T val) {
#if __CUDA_ARCH__ >= 800
  return __reduce_add_sync(0xffffffff, val);
#else
#pragma unroll
  for (int mask = Width / 2; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask, kWarpSize);
  }

  return val;
#endif
}

} // namespace

#endif // MULTIBYTE_ANS_INCLUDE_UTILS_PTXUTILS_H_
