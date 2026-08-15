#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "coccl_pipeline_layout.h"

namespace {

constexpr unsigned char kGuard = 0xa5;
constexpr size_t kGuardBytes = 64;

void checkCuda(cudaError_t result, const char* call, int line) {
  if (result == cudaSuccess) return;
  std::fprintf(stderr, "line %d: %s: %s\n", line, call,
               cudaGetErrorString(result));
  std::exit(1);
}

void checkNccl(ncclResult_t result, const char* call, int line) {
  if (result == ncclSuccess) return;
  std::fprintf(stderr, "line %d: %s: %d\n", line, call, (int)result);
  std::exit(1);
}

#define CUDA_CHECK(call) checkCuda((call), #call, __LINE__)
#define NCCL_CHECK(call) checkNccl((call), #call, __LINE__)

struct LayoutCase {
  size_t sliceBytes;
  size_t chunks;
  size_t padding;
  size_t sourceOffset;
  size_t packedOffset;
};

unsigned char value(size_t row, size_t column) {
  return (unsigned char)((row * 131 + column * 17 + 29) & 0xff);
}

bool unchanged(const std::vector<unsigned char>& data, size_t begin,
               size_t end) {
  for (size_t i = begin; i < end; ++i) {
    if (data[i] != kGuard) return false;
  }
  return true;
}

bool runCase(const LayoutCase& test) {
  const size_t pitch = test.sliceBytes + test.padding;
  const size_t pitchedSpan =
      (test.chunks - 1) * pitch + test.sliceBytes;
  const size_t packedBytes = test.sliceBytes * test.chunks;
  const size_t sourceBytes =
      kGuardBytes + test.sourceOffset + pitchedSpan + kGuardBytes;
  const size_t packedAllocation =
      kGuardBytes + test.packedOffset + packedBytes + kGuardBytes;

  std::vector<unsigned char> source(sourceBytes, kGuard);
  std::vector<unsigned char> packed(packedAllocation, kGuard);
  std::vector<unsigned char> unpacked(sourceBytes, kGuard);
  unsigned char* sourceData =
      source.data() + kGuardBytes + test.sourceOffset;
  for (size_t row = 0; row < test.chunks; ++row) {
    for (size_t column = 0; column < test.sliceBytes; ++column) {
      sourceData[row * pitch + column] = value(row, column);
    }
  }

  unsigned char* deviceSource = nullptr;
  unsigned char* devicePacked = nullptr;
  unsigned char* deviceUnpacked = nullptr;
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaMalloc(&deviceSource, sourceBytes));
  CUDA_CHECK(cudaMalloc(&devicePacked, packedAllocation));
  CUDA_CHECK(cudaMalloc(&deviceUnpacked, sourceBytes));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaMemcpyAsync(deviceSource, source.data(), sourceBytes,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(devicePacked, packed.data(), packedAllocation,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(deviceUnpacked, unpacked.data(), sourceBytes,
                             cudaMemcpyHostToDevice, stream));

  const unsigned char* deviceSourceData =
      deviceSource + kGuardBytes + test.sourceOffset;
  unsigned char* devicePackedData =
      devicePacked + kGuardBytes + test.packedOffset;
  unsigned char* deviceUnpackedData =
      deviceUnpacked + kGuardBytes + test.sourceOffset;
  NCCL_CHECK(cocclLaunchPackSlice(deviceSourceData, pitch, devicePackedData,
                                  test.sliceBytes, test.chunks, stream));
  NCCL_CHECK(cocclLaunchUnpackSlice(devicePackedData, deviceUnpackedData,
                                    pitch, test.sliceBytes, test.chunks,
                                    stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(packed.data(), devicePacked, packedAllocation,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(unpacked.data(), deviceUnpacked, sourceBytes,
                        cudaMemcpyDeviceToHost));

  bool passed = true;
  const unsigned char* packedData =
      packed.data() + kGuardBytes + test.packedOffset;
  const unsigned char* unpackedData =
      unpacked.data() + kGuardBytes + test.sourceOffset;
  for (size_t row = 0; row < test.chunks && passed; ++row) {
    for (size_t column = 0; column < test.sliceBytes; ++column) {
      const unsigned char expected = value(row, column);
      if (packedData[row * test.sliceBytes + column] != expected ||
          unpackedData[row * pitch + column] != expected) {
        passed = false;
        break;
      }
    }
    if (row + 1 < test.chunks) {
      passed &= unchanged(unpacked,
                          kGuardBytes + test.sourceOffset + row * pitch +
                              test.sliceBytes,
                          kGuardBytes + test.sourceOffset + (row + 1) * pitch);
    }
  }
  passed &= unchanged(packed, 0, kGuardBytes + test.packedOffset);
  passed &= unchanged(packed,
                      kGuardBytes + test.packedOffset + packedBytes,
                      packed.size());
  passed &= unchanged(unpacked, 0, kGuardBytes + test.sourceOffset);
  passed &= unchanged(unpacked,
                      kGuardBytes + test.sourceOffset + pitchedSpan,
                      unpacked.size());

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(deviceUnpacked));
  CUDA_CHECK(cudaFree(devicePacked));
  CUDA_CHECK(cudaFree(deviceSource));
  return passed;
}

float measure(bool pack, const unsigned char* source,
              unsigned char* destination, size_t chunkBytes,
              size_t sliceBytes, size_t chunks, int depth,
              cudaStream_t stream) {
  constexpr int warmup = 5;
  constexpr int iterations = 20;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  auto launch = [&]() {
    for (int slice = 0; slice < depth; ++slice) {
      if (pack) {
        NCCL_CHECK(cocclLaunchPackSlice(
            source + (size_t)slice * sliceBytes, chunkBytes, destination,
            sliceBytes, chunks, stream));
      } else {
        NCCL_CHECK(cocclLaunchUnpackSlice(
            source, destination + (size_t)slice * sliceBytes, chunkBytes,
            sliceBytes, chunks, stream));
      }
    }
  };
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < iterations; ++i) launch();
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsedMs = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));
  return elapsedMs * 1000.0f / iterations;
}

void benchmark(size_t bytes, int depth) {
  constexpr size_t chunks = 4;
  if (depth == 1) {
    std::printf("bytes,chunks,depth,pack_us,unpack_us,layout_us\n");
    std::printf("%zu,%zu,%d,0,0,0\n", bytes, chunks, depth);
    return;
  }
  const size_t chunkBytes = bytes / chunks;
  const size_t sliceBytes = chunkBytes / (size_t)depth;
  const size_t packedBytes = sliceBytes * chunks;
  unsigned char* raw = nullptr;
  unsigned char* packed = nullptr;
  unsigned char* output = nullptr;
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaMalloc(&raw, bytes));
  CUDA_CHECK(cudaMalloc(&packed, packedBytes));
  CUDA_CHECK(cudaMalloc(&output, bytes));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaMemsetAsync(raw, 0x5a, bytes, stream));
  const float packUs = measure(true, raw, packed, chunkBytes, sliceBytes,
                               chunks, depth, stream);
  const float unpackUs = measure(false, packed, output, chunkBytes,
                                 sliceBytes, chunks, depth, stream);
  std::printf("bytes,chunks,depth,pack_us,unpack_us,layout_us\n");
  std::printf("%zu,%zu,%d,%.6f,%.6f,%.6f\n", bytes, chunks, depth,
              packUs, unpackUs, packUs + unpackUs);
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(output));
  CUDA_CHECK(cudaFree(packed));
  CUDA_CHECK(cudaFree(raw));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 4 && std::strcmp(argv[1], "--benchmark") == 0) {
    benchmark(std::strtoull(argv[2], nullptr, 10), std::atoi(argv[3]));
    return 0;
  }
  const LayoutCase cases[] = {
      {1, 1, 3, 0, 0},       {3, 4, 5, 1, 3},
      {16, 4, 16, 0, 0},     {17, 4, 15, 0, 0},
      {257, 4, 255, 0, 0},   {65535, 4, 65537, 1, 3},
  };
  for (const LayoutCase& test : cases) {
    if (!runCase(test)) return 1;
  }
  std::printf("coccl M5 pipeline layout: PASS (%zu cases)\n",
              sizeof(cases) / sizeof(cases[0]));
  return 0;
}
