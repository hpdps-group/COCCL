#include "coccl_reduce_helper.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr int kReduceThreads = 256;
constexpr size_t kMaxReduceBlocks = 4096;

template <typename T>
__device__ __forceinline__ float cocclToFloat(T value);

template <>
__device__ __forceinline__ float cocclToFloat(float value) {
  return value;
}

template <>
__device__ __forceinline__ float cocclToFloat(__half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float cocclToFloat(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T cocclFromFloat(float value);

template <>
__device__ __forceinline__ float cocclFromFloat(float value) {
  return value;
}

template <>
__device__ __forceinline__ __half cocclFromFloat(float value) {
  return __float2half(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 cocclFromFloat(float value) {
  return __float2bfloat16(value);
}

template <typename T, int Elements>
struct alignas(sizeof(T) * Elements) cocclAlignedVector {
  T values[Elements];
};

template <typename T, int Elements>
__global__ __launch_bounds__(kReduceThreads) void cocclAddKernel(
    const T* input1, const T* input2, T* output, size_t vectorCount) {
  using Vector = cocclAlignedVector<T, Elements>;
  const Vector* vectorInput1 = reinterpret_cast<const Vector*>(input1);
  const Vector* vectorInput2 = reinterpret_cast<const Vector*>(input2);
  Vector* vectorOutput = reinterpret_cast<Vector*>(output);
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

  for (size_t index =
           static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < vectorCount; index += stride) {
    const Vector left = vectorInput1[index];
    const Vector right = vectorInput2[index];
    Vector result;
#pragma unroll
    for (int element = 0; element < Elements; ++element) {
      result.values[element] = cocclFromFloat<T>(
          cocclToFloat(left.values[element]) +
          cocclToFloat(right.values[element]));
    }
    vectorOutput[index] = result;
  }
}

template <typename T, int Elements, int StaticChunks>
__global__ __launch_bounds__(kReduceThreads) void cocclReduceChunksKernel(
    const T* input, size_t chunkCount, T* output, size_t vectorCount,
    int runtimeChunks) {
  // Input is chunk-major. Threads walk adjacent output vectors, so every load
  // from a given chunk is coalesced; each thread keeps its partial sums in
  // registers and writes only after reading all chunks. The latter also makes
  // output == input safe without shared memory or block-wide synchronization.
  using Vector = cocclAlignedVector<T, Elements>;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

  for (size_t vectorIndex =
           static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       vectorIndex < vectorCount; vectorIndex += stride) {
    float sums[Elements] = {};

    if constexpr (StaticChunks > 0) {
#pragma unroll
      for (int chunk = 0; chunk < StaticChunks; ++chunk) {
        const T* chunkBase = input + static_cast<size_t>(chunk) * chunkCount;
        const Vector value =
            reinterpret_cast<const Vector*>(chunkBase)[vectorIndex];
#pragma unroll
        for (int element = 0; element < Elements; ++element) {
          sums[element] += cocclToFloat(value.values[element]);
        }
      }
    } else {
      for (int chunk = 0; chunk < runtimeChunks; ++chunk) {
        const T* chunkBase = input + static_cast<size_t>(chunk) * chunkCount;
        const Vector value =
            reinterpret_cast<const Vector*>(chunkBase)[vectorIndex];
#pragma unroll
        for (int element = 0; element < Elements; ++element) {
          sums[element] += cocclToFloat(value.values[element]);
        }
      }
    }

    Vector result;
#pragma unroll
    for (int element = 0; element < Elements; ++element) {
      result.values[element] = cocclFromFloat<T>(sums[element]);
    }
    reinterpret_cast<Vector*>(output)[vectorIndex] = result;
  }
}

int cocclReduceGrid(size_t workItems) {
  size_t blocks =
      (workItems + static_cast<size_t>(kReduceThreads) - 1) /
      static_cast<size_t>(kReduceThreads);
  if (blocks > kMaxReduceBlocks) blocks = kMaxReduceBlocks;
  return static_cast<int>(blocks == 0 ? 1 : blocks);
}

ncclResult_t cocclCheckLaunch() {
  return cudaGetLastError() == cudaSuccess ? ncclSuccess
                                           : ncclUnhandledCudaError;
}

template <typename T, int Elements>
ncclResult_t cocclLaunchAddTyped(const void* input1, const void* input2,
                                 void* output, size_t count,
                                 cudaStream_t stream) {
  const size_t vectorCount = count / Elements;
  cocclAddKernel<T, Elements>
      <<<cocclReduceGrid(vectorCount), kReduceThreads, 0, stream>>>(
          static_cast<const T*>(input1), static_cast<const T*>(input2),
          static_cast<T*>(output), vectorCount);
  return cocclCheckLaunch();
}

template <typename T, int Elements, int StaticChunks>
ncclResult_t cocclLaunchReduceChunksKernel(const void* input,
                                           size_t chunkCount, void* output,
                                           int numChunks,
                                           cudaStream_t stream) {
  const size_t vectorCount = chunkCount / Elements;
  cocclReduceChunksKernel<T, Elements, StaticChunks>
      <<<cocclReduceGrid(vectorCount), kReduceThreads, 0, stream>>>(
          static_cast<const T*>(input), chunkCount, static_cast<T*>(output),
          vectorCount, numChunks);
  return cocclCheckLaunch();
}

template <typename T, int Elements>
ncclResult_t cocclLaunchReduceChunksSpecialized(const void* input,
                                                size_t chunkCount,
                                                void* output, int numChunks,
                                                cudaStream_t stream) {
  switch (numChunks) {
    case 2:
      return cocclLaunchReduceChunksKernel<T, Elements, 2>(
          input, chunkCount, output, numChunks, stream);
    case 4:
      return cocclLaunchReduceChunksKernel<T, Elements, 4>(
          input, chunkCount, output, numChunks, stream);
    case 8:
      return cocclLaunchReduceChunksKernel<T, Elements, 8>(
          input, chunkCount, output, numChunks, stream);
    default:
      // Fully unrolling 16 or more FP16/BF16 chunks significantly raises
      // register pressure. The runtime loop keeps occupancy high while the
      // memory accesses remain coalesced.
      return cocclLaunchReduceChunksKernel<T, Elements, 0>(
          input, chunkCount, output, numChunks, stream);
  }
}

size_t cocclVectorAlignment(const void* input1, const void* input2,
                            const void* output, size_t rowBytes) {
  const uintptr_t alignment = reinterpret_cast<uintptr_t>(input1) |
      reinterpret_cast<uintptr_t>(input2) |
      reinterpret_cast<uintptr_t>(output) | rowBytes;
  if ((alignment & 15) == 0) return 16;
  if ((alignment & 7) == 0) return 8;
  if ((alignment & 3) == 0) return 4;
  return 1;
}

template <typename T>
ncclResult_t cocclLaunchAdd(const void* input1, const void* input2,
                            void* output, size_t count,
                            cudaStream_t stream) {
  const size_t bytes = count * sizeof(T);
  switch (cocclVectorAlignment(input1, input2, output, bytes)) {
    case 16:
      return cocclLaunchAddTyped<T, 16 / sizeof(T)>(
          input1, input2, output, count, stream);
    case 8:
      return cocclLaunchAddTyped<T, 8 / sizeof(T)>(
          input1, input2, output, count, stream);
    case 4:
      return cocclLaunchAddTyped<T, 4 / sizeof(T)>(
          input1, input2, output, count, stream);
    default:
      return cocclLaunchAddTyped<T, 1>(input1, input2, output, count,
                                       stream);
  }
}

template <typename T>
ncclResult_t cocclLaunchReduceChunks(const void* input, size_t chunkCount,
                                     void* output, int numChunks,
                                     cudaStream_t stream) {
  const size_t chunkBytes = chunkCount * sizeof(T);
  switch (cocclVectorAlignment(input, nullptr, output, chunkBytes)) {
    case 16:
      return cocclLaunchReduceChunksSpecialized<T, 16 / sizeof(T)>(
          input, chunkCount, output, numChunks, stream);
    case 8:
      return cocclLaunchReduceChunksSpecialized<T, 8 / sizeof(T)>(
          input, chunkCount, output, numChunks, stream);
    case 4:
      return cocclLaunchReduceChunksSpecialized<T, 4 / sizeof(T)>(
          input, chunkCount, output, numChunks, stream);
    default:
      return cocclLaunchReduceChunksSpecialized<T, 1>(
          input, chunkCount, output, numChunks, stream);
  }
}

}  // namespace

ncclResult_t cocclLaunchReductionColl(const void* input1, const void* input2,
                                      void* output,
                                      ncclDataType_t datatype,
                                      size_t inputCount,
                                      cudaStream_t stream) {
  if (inputCount == 0) return ncclSuccess;
  if (input1 == nullptr || input2 == nullptr || output == nullptr) {
    return ncclInvalidArgument;
  }

  switch (datatype) {
    case ncclFloat16:
      return cocclLaunchAdd<__half>(input1, input2, output, inputCount,
                                    stream);
    case ncclFloat32:
      return cocclLaunchAdd<float>(input1, input2, output, inputCount, stream);
    case ncclBfloat16:
      return cocclLaunchAdd<__nv_bfloat16>(input1, input2, output, inputCount,
                                           stream);
    default:
      return ncclInvalidArgument;
  }
}

ncclResult_t cocclLaunchReduceChunk(const void* input, size_t chunkCount,
                                    void* output, ncclDataType_t datatype,
                                    int numChunks, cudaStream_t stream) {
  if (chunkCount == 0) return ncclSuccess;
  if (input == nullptr || output == nullptr || numChunks <= 0) {
    return ncclInvalidArgument;
  }

  size_t typeBytes = 0;
  switch (datatype) {
    case ncclFloat16:
    case ncclBfloat16:
      typeBytes = 2;
      break;
    case ncclFloat32:
      typeBytes = 4;
      break;
    default:
      return ncclInvalidArgument;
  }
  if (chunkCount > std::numeric_limits<size_t>::max() / typeBytes) {
    return ncclInvalidArgument;
  }

  if (numChunks == 1) {
    if (input == output) return ncclSuccess;
    return cudaMemcpyAsync(output, input, chunkCount * typeBytes,
                           cudaMemcpyDeviceToDevice, stream) == cudaSuccess
        ? ncclSuccess
        : ncclUnhandledCudaError;
  }

  switch (datatype) {
    case ncclFloat16:
      return cocclLaunchReduceChunks<__half>(input, chunkCount, output,
                                             numChunks, stream);
    case ncclFloat32:
      return cocclLaunchReduceChunks<float>(input, chunkCount, output,
                                            numChunks, stream);
    case ncclBfloat16:
      return cocclLaunchReduceChunks<__nv_bfloat16>(
          input, chunkCount, output, numChunks, stream);
    default:
      return ncclInvalidArgument;
  }
}
