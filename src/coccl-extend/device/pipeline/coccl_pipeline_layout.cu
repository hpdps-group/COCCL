#include "coccl_pipeline_layout.h"

#include <cuda_runtime.h>

#include <cstdint>

#if CUDART_VERSION < 12040
#error "COCCL pipeline layout kernels require CUDA Toolkit 12.4 or newer"
#endif

namespace {

constexpr int kLayoutThreads = 256;
constexpr unsigned int kMaxLayoutBlocks = 256 * 1024;
constexpr size_t kTargetBytesPerBlock = 4 * 1024;
constexpr unsigned int kMaxGridY = 65535;

template <typename CopyType, int Unroll>
struct cocclLayoutRowCopier {
  __device__ __forceinline__ static void copy(
      const CopyType* __restrict__ source,
      CopyType* __restrict__ destination, size_t elementsPerChunk) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    size_t element =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t unrolledOffset = (Unroll - 1) * stride;
    for (; element < elementsPerChunk &&
           unrolledOffset < elementsPerChunk - element;
         element += (size_t)Unroll * stride) {
      CopyType values[Unroll];
#pragma unroll
      for (int i = 0; i < Unroll; ++i) {
        values[i] = source[element + (size_t)i * stride];
      }
#pragma unroll
      for (int i = 0; i < Unroll; ++i) {
        destination[element + (size_t)i * stride] = values[i];
      }
    }
    for (; element < elementsPerChunk; element += stride) {
      destination[element] = source[element];
    }
  }
};

template <typename CopyType>
struct cocclLayoutRowCopier<CopyType, 1> {
  __device__ __forceinline__ static void copy(
      const CopyType* __restrict__ source,
      CopyType* __restrict__ destination, size_t elementsPerChunk) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t element =
             static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         element < elementsPerChunk; element += stride) {
      destination[element] = source[element];
    }
  }
};

template <typename CopyType>
__device__ __forceinline__ void cocclCopyLayoutRow(
    const CopyType* __restrict__ source, CopyType* __restrict__ destination,
    size_t elementsPerChunk) {
  constexpr int kUnroll =
      kTargetBytesPerBlock / (kLayoutThreads * sizeof(CopyType));
  cocclLayoutRowCopier<CopyType, kUnroll>::copy(
      source, destination, elementsPerChunk);
}

template <typename CopyType>
__global__ __launch_bounds__(kLayoutThreads) void cocclPackSliceKernel(
    const unsigned char* __restrict__ source, size_t sourcePitchBytes,
    unsigned char* __restrict__ destination, size_t sliceBytes,
    size_t chunkCount) {
  const size_t elementsPerChunk = sliceBytes / sizeof(CopyType);
  for (size_t chunk = blockIdx.y; chunk < chunkCount; chunk += gridDim.y) {
    const CopyType* __restrict__ sourceRow =
        reinterpret_cast<const CopyType*>(source + chunk * sourcePitchBytes);
    CopyType* __restrict__ destinationRow = reinterpret_cast<CopyType*>(
        destination + chunk * sliceBytes);
    cocclCopyLayoutRow(sourceRow, destinationRow, elementsPerChunk);
  }
}

template <typename CopyType>
__global__ __launch_bounds__(kLayoutThreads) void cocclUnpackSliceKernel(
    const unsigned char* __restrict__ source,
    unsigned char* __restrict__ destination, size_t destinationPitchBytes,
    size_t sliceBytes, size_t chunkCount) {
  const size_t elementsPerChunk = sliceBytes / sizeof(CopyType);
  for (size_t chunk = blockIdx.y; chunk < chunkCount; chunk += gridDim.y) {
    const CopyType* __restrict__ sourceRow = reinterpret_cast<const CopyType*>(
        source + chunk * sliceBytes);
    CopyType* __restrict__ destinationRow = reinterpret_cast<CopyType*>(
        destination + chunk * destinationPitchBytes);
    cocclCopyLayoutRow(sourceRow, destinationRow, elementsPerChunk);
  }
}

template <typename CopyType>
dim3 cocclLayoutGrid(size_t sliceBytes, size_t chunkCount) {
  const size_t elementsPerChunk = sliceBytes / sizeof(CopyType);
  const size_t neededBlocks =
      (elementsPerChunk + kLayoutThreads - 1) / kLayoutThreads;
  const size_t targetBlocks =
      (sliceBytes + kTargetBytesPerBlock - 1) / kTargetBytesPerBlock;
  const unsigned int gridY = static_cast<unsigned int>(
      chunkCount < kMaxGridY ? chunkCount : kMaxGridY);
  unsigned int blocksPerRow =
      gridY == 0 ? 1 : kMaxLayoutBlocks / gridY;
  // Keep roughly 4 KiB of payload behind each block. This exposes enough
  // independent memory work for large transfers without over-launching the
  // narrower fallback kernels used for partially aligned layouts.
  if (blocksPerRow == 0) blocksPerRow = 1;
  if (neededBlocks < blocksPerRow) {
    blocksPerRow = static_cast<unsigned int>(neededBlocks);
  }
  if (targetBlocks < blocksPerRow) {
    blocksPerRow = static_cast<unsigned int>(targetBlocks);
  }
  return dim3(blocksPerRow == 0 ? 1 : blocksPerRow, gridY, 1);
}

template <typename CopyType>
ncclResult_t cocclLaunchPackTyped(const void* source,
                                  size_t sourcePitchBytes, void* destination,
                                  size_t sliceBytes, size_t chunkCount,
                                  cudaStream_t stream) {
  dim3 grid = cocclLayoutGrid<CopyType>(sliceBytes, chunkCount);
  cocclPackSliceKernel<CopyType><<<grid, kLayoutThreads, 0, stream>>>(
      static_cast<const unsigned char*>(source), sourcePitchBytes,
      static_cast<unsigned char*>(destination), sliceBytes, chunkCount);
  return cudaGetLastError() == cudaSuccess ? ncclSuccess
                                           : ncclUnhandledCudaError;
}

template <typename CopyType>
ncclResult_t cocclLaunchUnpackTyped(const void* source, void* destination,
                                    size_t destinationPitchBytes,
                                    size_t sliceBytes, size_t chunkCount,
                                    cudaStream_t stream) {
  dim3 grid = cocclLayoutGrid<CopyType>(sliceBytes, chunkCount);
  cocclUnpackSliceKernel<CopyType><<<grid, kLayoutThreads, 0, stream>>>(
      static_cast<const unsigned char*>(source),
      static_cast<unsigned char*>(destination), destinationPitchBytes,
      sliceBytes, chunkCount);
  return cudaGetLastError() == cudaSuccess ? ncclSuccess
                                           : ncclUnhandledCudaError;
}

size_t cocclLayoutVectorBytes(const void* source, size_t sourcePitchBytes,
                              const void* destination,
                              size_t destinationPitchBytes,
                              size_t sliceBytes) {
  const uintptr_t alignment = reinterpret_cast<uintptr_t>(source) |
      reinterpret_cast<uintptr_t>(destination) | sourcePitchBytes |
      destinationPitchBytes | sliceBytes;
  if ((alignment & 15) == 0) return 16;
  if ((alignment & 7) == 0) return 8;
  if ((alignment & 3) == 0) return 4;
  return 1;
}

}  // namespace

ncclResult_t cocclLaunchPackSlice(const void* source,
                                  size_t sourcePitchBytes, void* destination,
                                  size_t sliceBytes, size_t chunkCount,
                                  cudaStream_t stream) {
  if (source == nullptr || destination == nullptr || sliceBytes == 0 ||
      chunkCount == 0 || sourcePitchBytes < sliceBytes) {
    return ncclInvalidArgument;
  }
  switch (cocclLayoutVectorBytes(source, sourcePitchBytes, destination,
                                 sliceBytes, sliceBytes)) {
    case 16:
      return cocclLaunchPackTyped<uint4>(source, sourcePitchBytes, destination,
                                         sliceBytes, chunkCount, stream);
    case 8:
      return cocclLaunchPackTyped<unsigned long long>(
          source, sourcePitchBytes, destination, sliceBytes, chunkCount,
          stream);
    case 4:
      return cocclLaunchPackTyped<unsigned int>(source, sourcePitchBytes,
                                                 destination, sliceBytes,
                                                 chunkCount, stream);
    default:
      return cocclLaunchPackTyped<unsigned char>(
          source, sourcePitchBytes, destination, sliceBytes, chunkCount,
          stream);
  }
}

ncclResult_t cocclLaunchUnpackSlice(const void* source, void* destination,
                                    size_t destinationPitchBytes,
                                    size_t sliceBytes, size_t chunkCount,
                                    cudaStream_t stream) {
  if (source == nullptr || destination == nullptr || sliceBytes == 0 ||
      chunkCount == 0 || destinationPitchBytes < sliceBytes) {
    return ncclInvalidArgument;
  }
  switch (cocclLayoutVectorBytes(source, sliceBytes, destination,
                                 destinationPitchBytes, sliceBytes)) {
    case 16:
      return cocclLaunchUnpackTyped<uint4>(
          source, destination, destinationPitchBytes, sliceBytes, chunkCount,
          stream);
    case 8:
      return cocclLaunchUnpackTyped<unsigned long long>(
          source, destination, destinationPitchBytes, sliceBytes, chunkCount,
          stream);
    case 4:
      return cocclLaunchUnpackTyped<unsigned int>(
          source, destination, destinationPitchBytes, sliceBytes, chunkCount,
          stream);
    default:
      return cocclLaunchUnpackTyped<unsigned char>(
          source, destination, destinationPitchBytes, sliceBytes, chunkCount,
          stream);
  }
}
