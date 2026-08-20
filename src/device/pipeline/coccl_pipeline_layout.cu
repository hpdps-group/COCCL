#include "coccl_pipeline_layout.h"

#include <cuda_runtime.h>

#include <cstdint>

#if CUDART_VERSION < 12040
#error "COCCL pipeline layout kernels require CUDA Toolkit 12.4 or newer"
#endif

namespace {

constexpr int kLayoutThreads = 256;
constexpr unsigned int kMaxLayoutBlocks = 32 * 1024;
constexpr size_t kTargetBytesPerBlock = 32 * 1024;
constexpr unsigned int kMaxGridY = 65535;

template <typename CopyType, int Unroll>
struct LayoutRowCopier {
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
#pragma unroll
      for (int i = 0; i < Unroll; ++i) {
        destination[element + (size_t)i * stride] =
            source[element + (size_t)i * stride];
      }
    }
    for (; element < elementsPerChunk; element += stride) {
      destination[element] = source[element];
    }
  }
};

template <typename CopyType>
struct LayoutRowCopier<CopyType, 1> {
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
__device__ __forceinline__ void copyLayoutRow(
    const CopyType* __restrict__ source, CopyType* __restrict__ destination,
    size_t elementsPerChunk) {
  constexpr int kUnroll =
      kTargetBytesPerBlock / (kLayoutThreads * sizeof(CopyType));
  LayoutRowCopier<CopyType, kUnroll>::copy(source, destination,
                                            elementsPerChunk);
}

template <typename CopyType, bool Swizzle>
__global__ __launch_bounds__(kLayoutThreads) void packSliceKernel(
    const unsigned char* __restrict__ source, size_t sourcePitchBytes,
    unsigned char* __restrict__ destination, size_t sliceBytes,
    size_t chunkCount, int nNodes, int ranksPerNode) {
  const size_t elementsPerChunk = sliceBytes / sizeof(CopyType);
  for (size_t sourceChunk = blockIdx.y; sourceChunk < chunkCount;
       sourceChunk += gridDim.y) {
    const size_t destinationChunk = Swizzle
        ? (sourceChunk % (size_t)ranksPerNode) * (size_t)nNodes +
              sourceChunk / (size_t)ranksPerNode
        : sourceChunk;
    const CopyType* __restrict__ sourceRow =
        reinterpret_cast<const CopyType*>(
            source + sourceChunk * sourcePitchBytes);
    CopyType* __restrict__ destinationRow = reinterpret_cast<CopyType*>(
        destination + destinationChunk * sliceBytes);
    copyLayoutRow(sourceRow, destinationRow, elementsPerChunk);
  }
}

template <typename CopyType>
__global__ __launch_bounds__(kLayoutThreads) void unpackSliceKernel(
    const unsigned char* __restrict__ source,
    unsigned char* __restrict__ destination, size_t destinationPitchBytes,
    size_t sliceBytes, size_t chunkCount) {
  const size_t elementsPerChunk = sliceBytes / sizeof(CopyType);
  for (size_t chunk = blockIdx.y; chunk < chunkCount; chunk += gridDim.y) {
    const CopyType* __restrict__ sourceRow = reinterpret_cast<const CopyType*>(
        source + chunk * sliceBytes);
    CopyType* __restrict__ destinationRow = reinterpret_cast<CopyType*>(
        destination + chunk * destinationPitchBytes);
    copyLayoutRow(sourceRow, destinationRow, elementsPerChunk);
  }
}

template <typename CopyType>
dim3 layoutGrid(size_t sliceBytes, size_t chunkCount) {
  const size_t elementsPerChunk = sliceBytes / sizeof(CopyType);
  const size_t neededBlocks =
      (elementsPerChunk + kLayoutThreads - 1) / kLayoutThreads;
  const size_t targetBlocks =
      (sliceBytes + kTargetBytesPerBlock - 1) / kTargetBytesPerBlock;
  const unsigned int gridY = static_cast<unsigned int>(
      chunkCount < kMaxGridY ? chunkCount : kMaxGridY);
  unsigned int blocksPerRow = kMaxLayoutBlocks / gridY;
  if (blocksPerRow == 0) blocksPerRow = 1;
  if (neededBlocks < blocksPerRow) {
    blocksPerRow = static_cast<unsigned int>(neededBlocks);
  }
  if (targetBlocks < blocksPerRow) {
    blocksPerRow = static_cast<unsigned int>(targetBlocks);
  }
  return dim3(blocksPerRow == 0 ? 1 : blocksPerRow, gridY, 1);
}

template <typename CopyType, bool Swizzle>
ncclResult_t launchPackTyped(const void* source, size_t sourcePitchBytes,
                             void* destination, size_t sliceBytes,
                             size_t chunkCount, int nNodes,
                             int ranksPerNode, cudaStream_t stream) {
  packSliceKernel<CopyType, Swizzle>
      <<<layoutGrid<CopyType>(sliceBytes, chunkCount), kLayoutThreads, 0,
         stream>>>(
      static_cast<const unsigned char*>(source), sourcePitchBytes,
      static_cast<unsigned char*>(destination), sliceBytes, chunkCount,
      nNodes, ranksPerNode);
  return cudaGetLastError() == cudaSuccess ? ncclSuccess
                                           : ncclUnhandledCudaError;
}

size_t layoutVectorBytes(const void* source, size_t sourcePitchBytes,
                         const void* destination,
                         size_t destinationPitchBytes, size_t sliceBytes);

template <bool Swizzle>
ncclResult_t dispatchPack(const void* source, size_t sourcePitchBytes,
                          void* destination, size_t sliceBytes,
                          size_t chunkCount, int nNodes, int ranksPerNode,
                          cudaStream_t stream) {
  switch (layoutVectorBytes(source, sourcePitchBytes, destination, sliceBytes,
                            sliceBytes)) {
    case 16:
      return launchPackTyped<uint4, Swizzle>(
          source, sourcePitchBytes, destination, sliceBytes, chunkCount,
          nNodes, ranksPerNode, stream);
    case 8:
      return launchPackTyped<unsigned long long, Swizzle>(
          source, sourcePitchBytes, destination, sliceBytes, chunkCount,
          nNodes, ranksPerNode, stream);
    case 4:
      return launchPackTyped<unsigned int, Swizzle>(
          source, sourcePitchBytes, destination, sliceBytes, chunkCount,
          nNodes, ranksPerNode, stream);
    case 2:
      return launchPackTyped<unsigned short, Swizzle>(
          source, sourcePitchBytes, destination, sliceBytes, chunkCount,
          nNodes, ranksPerNode, stream);
    default:
      return launchPackTyped<unsigned char, Swizzle>(
          source, sourcePitchBytes, destination, sliceBytes, chunkCount,
          nNodes, ranksPerNode, stream);
  }
}

template <typename CopyType>
ncclResult_t launchUnpackTyped(const void* source, void* destination,
                               size_t destinationPitchBytes,
                               size_t sliceBytes, size_t chunkCount,
                               cudaStream_t stream) {
  unpackSliceKernel<CopyType><<<layoutGrid<CopyType>(sliceBytes, chunkCount),
                                kLayoutThreads, 0, stream>>>(
      static_cast<const unsigned char*>(source),
      static_cast<unsigned char*>(destination), destinationPitchBytes,
      sliceBytes, chunkCount);
  return cudaGetLastError() == cudaSuccess ? ncclSuccess
                                           : ncclUnhandledCudaError;
}

size_t layoutVectorBytes(const void* source, size_t sourcePitchBytes,
                         const void* destination,
                         size_t destinationPitchBytes, size_t sliceBytes) {
  const uintptr_t alignment = reinterpret_cast<uintptr_t>(source) |
      reinterpret_cast<uintptr_t>(destination) | sourcePitchBytes |
      destinationPitchBytes | sliceBytes;
  if ((alignment & 15) == 0) return 16;
  if ((alignment & 7) == 0) return 8;
  if ((alignment & 3) == 0) return 4;
  if ((alignment & 1) == 0) return 2;
  return 1;
}

}  // namespace

ncclResult_t cocclLaunchPackSlice(const void* source,
                                  size_t sourcePitchBytes, void* destination,
                                  size_t sliceBytes, size_t chunkCount,
                                  cocclPipelineInputLayout inputLayout,
                                  int nNodes, int ranksPerNode,
                                  cudaStream_t stream) {
  if (source == nullptr || destination == nullptr || sliceBytes == 0 ||
      chunkCount == 0 || sourcePitchBytes < sliceBytes) {
    return ncclInvalidArgument;
  }
  if (inputLayout == cocclPipelineInputContiguous &&
      layoutVectorBytes(source, sourcePitchBytes, destination, sliceBytes,
                        sliceBytes) < 16) {
    return cudaMemcpy2DAsync(
               destination, sliceBytes, source, sourcePitchBytes,
               sliceBytes, chunkCount, cudaMemcpyDeviceToDevice, stream) ==
            cudaSuccess
        ? ncclSuccess : ncclUnhandledCudaError;
  }
  return inputLayout == cocclPipelineInputHierarchicalSwizzle
      ? dispatchPack<true>(source, sourcePitchBytes, destination, sliceBytes,
                           chunkCount, nNodes, ranksPerNode, stream)
      : dispatchPack<false>(source, sourcePitchBytes, destination, sliceBytes,
                            chunkCount, nNodes, ranksPerNode, stream);
}

ncclResult_t cocclLaunchUnpackSlice(const void* source, void* destination,
                                    size_t destinationPitchBytes,
                                    size_t sliceBytes, size_t chunkCount,
                                    cudaStream_t stream) {
  if (source == nullptr || destination == nullptr || sliceBytes == 0 ||
      chunkCount == 0 || destinationPitchBytes < sliceBytes) {
    return ncclInvalidArgument;
  }
  if (layoutVectorBytes(source, sliceBytes, destination,
                        destinationPitchBytes, sliceBytes) < 16) {
    return cudaMemcpy2DAsync(
               destination, destinationPitchBytes, source, sliceBytes,
               sliceBytes, chunkCount, cudaMemcpyDeviceToDevice, stream) ==
            cudaSuccess
        ? ncclSuccess : ncclUnhandledCudaError;
  }
  return launchUnpackTyped<uint4>(source, destination, destinationPitchBytes,
                                  sliceBytes, chunkCount, stream);
}
