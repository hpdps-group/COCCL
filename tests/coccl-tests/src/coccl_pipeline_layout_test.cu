#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "pipeline/coccl_pipeline_layout.h"

namespace {

constexpr unsigned char kGuardValue = 0xa5;
constexpr unsigned char kDestinationValue = 0x3c;
constexpr size_t kGuardBytes = 64;

void checkCuda(cudaError_t result, const char* expression, int line) {
  if (result == cudaSuccess) return;
  std::fprintf(stderr, "CUDA failure at line %d: %s: %s\n", line,
               expression, cudaGetErrorString(result));
  std::exit(EXIT_FAILURE);
}

void checkNccl(ncclResult_t result, const char* expression, int line) {
  if (result == ncclSuccess) return;
  std::fprintf(stderr, "NCCL failure at line %d: %s: %d\n", line,
               expression, static_cast<int>(result));
  std::exit(EXIT_FAILURE);
}

#define CUDACHECK_TEST(call) checkCuda((call), #call, __LINE__)
#define NCCLCHECK_TEST(call) checkNccl((call), #call, __LINE__)

struct LayoutCase {
  size_t rowBytes;
  size_t rowCount;
  size_t rowPadding;
  size_t sourceOffset;
  size_t destinationOffset;
};

unsigned char pattern(size_t row, size_t column) {
  return static_cast<unsigned char>((row * 131 + column * 17 + 29) & 0xff);
}

bool checkGuard(const std::vector<unsigned char>& data, size_t begin,
                size_t end, const char* name) {
  for (size_t i = begin; i < end; ++i) {
    if (data[i] != kGuardValue) {
      std::fprintf(stderr, "%s guard changed at byte %zu\n", name, i);
      return false;
    }
  }
  return true;
}

bool runLayoutCase(const LayoutCase& testCase) {
  const size_t pitch = testCase.rowBytes + testCase.rowPadding;
  const size_t pitchedBytes =
      pitch * (testCase.rowCount - 1) + testCase.rowBytes;
  const size_t packedBytes = testCase.rowBytes * testCase.rowCount;
  const size_t sourceBytes =
      kGuardBytes + testCase.sourceOffset + pitchedBytes + kGuardBytes;
  const size_t packedAllocationBytes =
      kGuardBytes + testCase.destinationOffset + packedBytes + kGuardBytes;

  std::vector<unsigned char> source(sourceBytes, kGuardValue);
  std::vector<unsigned char> packed(packedAllocationBytes, kGuardValue);
  std::vector<unsigned char> unpacked(sourceBytes, kGuardValue);
  unsigned char* sourceData =
      source.data() + kGuardBytes + testCase.sourceOffset;
  for (size_t row = 0; row < testCase.rowCount; ++row) {
    std::fill(sourceData + row * pitch,
              sourceData + row * pitch + testCase.rowBytes,
              kDestinationValue);
    for (size_t column = 0; column < testCase.rowBytes; ++column) {
      sourceData[row * pitch + column] = pattern(row, column);
    }
  }

  unsigned char* deviceSource = nullptr;
  unsigned char* devicePacked = nullptr;
  unsigned char* deviceUnpacked = nullptr;
  cudaStream_t packStream = nullptr;
  cudaStream_t unpackStream = nullptr;
  cudaEvent_t packReady = nullptr;
  CUDACHECK_TEST(cudaMalloc(&deviceSource, sourceBytes));
  CUDACHECK_TEST(cudaMalloc(&devicePacked, packedAllocationBytes));
  CUDACHECK_TEST(cudaMalloc(&deviceUnpacked, sourceBytes));
  CUDACHECK_TEST(
      cudaStreamCreateWithFlags(&packStream, cudaStreamNonBlocking));
  CUDACHECK_TEST(
      cudaStreamCreateWithFlags(&unpackStream, cudaStreamNonBlocking));
  CUDACHECK_TEST(cudaEventCreateWithFlags(&packReady, cudaEventDefault));
  // Initialize on the Pack stream, then exercise the same event handoff used by
  // the overlap pipeline before Unpack consumes the packed layout.
  CUDACHECK_TEST(cudaMemcpyAsync(deviceSource, source.data(), sourceBytes,
                                 cudaMemcpyHostToDevice, packStream));
  CUDACHECK_TEST(cudaMemcpyAsync(devicePacked, packed.data(),
                                 packedAllocationBytes,
                                 cudaMemcpyHostToDevice, packStream));
  CUDACHECK_TEST(cudaMemcpyAsync(deviceUnpacked, unpacked.data(), sourceBytes,
                                 cudaMemcpyHostToDevice, packStream));

  const unsigned char* deviceSourceData =
      deviceSource + kGuardBytes + testCase.sourceOffset;
  unsigned char* devicePackedData =
      devicePacked + kGuardBytes + testCase.destinationOffset;
  unsigned char* deviceUnpackedData =
      deviceUnpacked + kGuardBytes + testCase.sourceOffset;
  NCCLCHECK_TEST(cocclLaunchPackSlice(
      deviceSourceData, pitch, devicePackedData, testCase.rowBytes,
      testCase.rowCount, packStream));
  CUDACHECK_TEST(cudaEventRecord(packReady, packStream));
  CUDACHECK_TEST(cudaStreamWaitEvent(unpackStream, packReady, 0));
  NCCLCHECK_TEST(cocclLaunchUnpackSlice(
      devicePackedData, deviceUnpackedData, pitch, testCase.rowBytes,
      testCase.rowCount, unpackStream));
  CUDACHECK_TEST(cudaStreamSynchronize(unpackStream));
  CUDACHECK_TEST(cudaMemcpy(packed.data(), devicePacked, packedAllocationBytes,
                            cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaMemcpy(unpacked.data(), deviceUnpacked, sourceBytes,
                            cudaMemcpyDeviceToHost));

  bool passed = true;
  const unsigned char* packedData =
      packed.data() + kGuardBytes + testCase.destinationOffset;
  const unsigned char* unpackedData =
      unpacked.data() + kGuardBytes + testCase.sourceOffset;
  for (size_t row = 0; row < testCase.rowCount; ++row) {
    for (size_t column = 0; column < testCase.rowBytes; ++column) {
      unsigned char expected = pattern(row, column);
      if (packedData[row * testCase.rowBytes + column] != expected) {
        std::fprintf(stderr,
                     "pack mismatch row=%zu column=%zu expected=%u got=%u\n",
                     row, column, expected,
                     packedData[row * testCase.rowBytes + column]);
        passed = false;
        goto done;
      }
      if (unpackedData[row * pitch + column] != expected) {
        std::fprintf(
            stderr,
            "unpack mismatch row=%zu column=%zu expected=%u got=%u "
            "rowBytes=%zu rows=%zu pitch=%zu srcOffset=%zu dstOffset=%zu\n",
            row, column, expected, unpackedData[row * pitch + column],
            testCase.rowBytes, testCase.rowCount, pitch,
            testCase.sourceOffset, testCase.destinationOffset);
        passed = false;
        goto done;
      }
    }
    if (row + 1 < testCase.rowCount) {
      for (size_t column = testCase.rowBytes; column < pitch; ++column) {
        if (unpackedData[row * pitch + column] != kGuardValue) {
          std::fprintf(stderr,
                       "unpack changed row padding row=%zu column=%zu\n", row,
                       column);
          passed = false;
          goto done;
        }
      }
    }
  }

done:
  passed &= checkGuard(packed, 0,
                       kGuardBytes + testCase.destinationOffset, "pack prefix");
  passed &= checkGuard(
      packed, kGuardBytes + testCase.destinationOffset + packedBytes,
      packed.size(), "pack suffix");
  passed &= checkGuard(unpacked, 0,
                       kGuardBytes + testCase.sourceOffset, "unpack prefix");
  passed &= checkGuard(
      unpacked, kGuardBytes + testCase.sourceOffset + pitchedBytes,
      unpacked.size(), "unpack suffix");

  CUDACHECK_TEST(cudaEventDestroy(packReady));
  CUDACHECK_TEST(cudaStreamDestroy(unpackStream));
  CUDACHECK_TEST(cudaStreamDestroy(packStream));
  CUDACHECK_TEST(cudaFree(deviceUnpacked));
  CUDACHECK_TEST(cudaFree(devicePacked));
  CUDACHECK_TEST(cudaFree(deviceSource));
  return passed;
}

enum BenchmarkOperation {
  BenchmarkPack,
  BenchmarkUnpack,
  BenchmarkMemcpy2DPack,
  BenchmarkMemcpy2DUnpack,
  BenchmarkMemcpyContiguous,
};

float measureOperation(BenchmarkOperation operation, const void* source,
                       size_t sourcePitch, void* destination,
                       size_t destinationPitch, size_t rowBytes,
                       size_t rowCount, int warmup, int iterations,
                       cudaStream_t stream) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDACHECK_TEST(cudaEventCreate(&start));
  CUDACHECK_TEST(cudaEventCreate(&stop));

  auto launch = [&]() {
    switch (operation) {
      case BenchmarkPack:
        NCCLCHECK_TEST(cocclLaunchPackSlice(source, sourcePitch, destination,
                                            rowBytes, rowCount, stream));
        break;
      case BenchmarkUnpack:
        NCCLCHECK_TEST(cocclLaunchUnpackSlice(source, destination,
                                              destinationPitch, rowBytes,
                                              rowCount, stream));
        break;
      case BenchmarkMemcpy2DPack:
      case BenchmarkMemcpy2DUnpack:
        CUDACHECK_TEST(cudaMemcpy2DAsync(destination, destinationPitch, source,
                                         sourcePitch, rowBytes, rowCount,
                                         cudaMemcpyDeviceToDevice, stream));
        break;
      case BenchmarkMemcpyContiguous:
        CUDACHECK_TEST(cudaMemcpyAsync(destination, source,
                                       rowBytes * rowCount,
                                       cudaMemcpyDeviceToDevice, stream));
        break;
    }
  };

  for (int i = 0; i < warmup; ++i) launch();
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  CUDACHECK_TEST(cudaEventRecord(start, stream));
  for (int i = 0; i < iterations; ++i) launch();
  CUDACHECK_TEST(cudaEventRecord(stop, stream));
  CUDACHECK_TEST(cudaEventSynchronize(stop));
  float elapsedMs = 0.0f;
  CUDACHECK_TEST(cudaEventElapsedTime(&elapsedMs, start, stop));
  CUDACHECK_TEST(cudaEventDestroy(stop));
  CUDACHECK_TEST(cudaEventDestroy(start));
  return elapsedMs / static_cast<float>(iterations);
}

void printBandwidth(const char* operation, size_t payloadBytes, size_t chunks,
                    float elapsedMs) {
  double seconds = static_cast<double>(elapsedMs) / 1000.0;
  double payloadGbPerSecond =
      static_cast<double>(payloadBytes) / seconds / 1.0e9;
  std::printf("%s,%zu,%zu,%.6f,%.3f,%.3f\n", operation, payloadBytes,
              chunks, elapsedMs * 1000.0f, payloadGbPerSecond,
              2.0 * payloadGbPerSecond);
}

void runBenchmark(bool unaligned) {
  const size_t sizes[] = {1024, 64 * 1024, 1024 * 1024,
                          64 * 1024 * 1024, 256 * 1024 * 1024,
                          1024ULL * 1024 * 1024};
  const size_t chunks[] = {1, 4, 8, 16};
  constexpr int warmup = 20;
  constexpr int iterations = 200;
  cudaStream_t stream = nullptr;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  std::printf("operation,payload_bytes,chunks,latency_us,payload_GBps,"
              "hbm_traffic_GBps\n");
  for (size_t payloadBytes : sizes) {
    for (size_t rowCount : chunks) {
      size_t rowBytes = std::max<size_t>(1, payloadBytes / rowCount);
      size_t actualPayloadBytes = rowBytes * rowCount;
      size_t sourceOffset = unaligned ? 1 : 0;
      size_t packedOffset = unaligned ? 3 : 0;
      size_t sourcePitch = rowCount == 1
          ? rowBytes
          : rowBytes * 2 + (unaligned ? 1 : 0);
      size_t sourceBytes = sourceOffset + sourcePitch * (rowCount - 1) +
          rowBytes + 16;
      size_t packedBytes = packedOffset + actualPayloadBytes + 16;
      unsigned char* source = nullptr;
      unsigned char* packed = nullptr;
      unsigned char* unpacked = nullptr;
      CUDACHECK_TEST(cudaMalloc(&source, sourceBytes));
      CUDACHECK_TEST(cudaMalloc(&packed, packedBytes));
      CUDACHECK_TEST(cudaMalloc(&unpacked, sourceBytes));
      CUDACHECK_TEST(cudaMemsetAsync(source, 0x5a, sourceBytes, stream));
      unsigned char* sourceData = source + sourceOffset;
      unsigned char* packedData = packed + packedOffset;
      unsigned char* unpackedData = unpacked + sourceOffset;

      float packMs = measureOperation(
          BenchmarkPack, sourceData, sourcePitch, packedData, rowBytes,
          rowBytes, rowCount, warmup, iterations, stream);
      float unpackMs = measureOperation(
          BenchmarkUnpack, packedData, rowBytes, unpackedData, sourcePitch,
          rowBytes, rowCount, warmup, iterations, stream);
      float memcpy2DPackMs = measureOperation(
          BenchmarkMemcpy2DPack, sourceData, sourcePitch, packedData, rowBytes,
          rowBytes, rowCount, warmup, iterations, stream);
      float memcpy2DUnpackMs = measureOperation(
          BenchmarkMemcpy2DUnpack, packedData, rowBytes, unpackedData,
          sourcePitch, rowBytes, rowCount, warmup, iterations, stream);
      float memcpyMs = measureOperation(
          BenchmarkMemcpyContiguous, packedData, rowBytes, unpackedData,
          rowBytes, rowBytes, rowCount, warmup, iterations, stream);

      printBandwidth("pack", actualPayloadBytes, rowCount, packMs);
      printBandwidth("unpack", actualPayloadBytes, rowCount, unpackMs);
      printBandwidth("memcpy2d_pack", actualPayloadBytes, rowCount,
                     memcpy2DPackMs);
      printBandwidth("memcpy2d_unpack", actualPayloadBytes, rowCount,
                     memcpy2DUnpackMs);
      printBandwidth("memcpy_contiguous", actualPayloadBytes, rowCount,
                     memcpyMs);

      CUDACHECK_TEST(cudaFree(unpacked));
      CUDACHECK_TEST(cudaFree(packed));
      CUDACHECK_TEST(cudaFree(source));
    }
  }
  CUDACHECK_TEST(cudaStreamDestroy(stream));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 &&
      (std::strcmp(argv[1], "--benchmark") == 0 ||
       std::strcmp(argv[1], "--benchmark-unaligned") == 0)) {
    runBenchmark(std::strcmp(argv[1], "--benchmark-unaligned") == 0);
    return EXIT_SUCCESS;
  }
  const LayoutCase cases[] = {
      {1, 1, 3, 0, 0},       {3, 4, 5, 1, 3},
      {7, 4, 9, 2, 5},       {15, 8, 17, 1, 7},
      {16, 4, 16, 0, 0},     {17, 4, 15, 0, 0},
      {31, 8, 33, 0, 0},     {64, 16, 64, 0, 0},
      {257, 4, 255, 0, 0},   {4096, 16, 4096, 0, 0},
      {65536, 4, 65536, 0, 0},
      {65535, 4, 65537, 1, 3},
  };
  const size_t caseCount = sizeof(cases) / sizeof(cases[0]);
  for (size_t i = 0; i < caseCount; ++i) {
    if (!runLayoutCase(cases[i])) return EXIT_FAILURE;
  }
  std::printf("coccl pipeline layout correctness: PASS (%zu cases)\n",
              caseCount);
  return EXIT_SUCCESS;
}
