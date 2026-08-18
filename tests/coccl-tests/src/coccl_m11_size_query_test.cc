#include "compress.h"

#include <cstdio>
#include <cstdlib>

namespace {

void fail(const char* expression, int line) {
  std::fprintf(stderr, "line %d: %s\n", line, expression);
  std::exit(1);
}

#define EXPECT(expression) \
  do { if (!(expression)) fail(#expression, __LINE__); } while (0)

enum EstimatorMode {
  EstimatorExact,
  EstimatorUnavailable,
  EstimatorZero,
  EstimatorUnaligned,
  EstimatorError,
};

EstimatorMode mode = EstimatorExact;
const void* expectedConfig = nullptr;
cocclCompressorSizeQuery observed = {};

ncclResult_t estimate(const cocclCompressorSizeQuery* query,
                      size_t* encodedBytes) {
  observed = *query;
  switch (mode) {
    case EstimatorExact:
      *encodedBytes = 1024;
      return ncclSuccess;
    case EstimatorUnavailable:
      return ncclInvalidUsage;
    case EstimatorZero:
      *encodedBytes = 0;
      return ncclSuccess;
    case EstimatorUnaligned:
      *encodedBytes = 1025;
      return ncclSuccess;
    case EstimatorError:
      return ncclSystemError;
  }
  return ncclInternalError;
}

void checkRawFallback() {
  cocclCompressorPlugin plugin = {};
  size_t bytes = 0;
  EXPECT(cocclQueryCompressorEncodedSizeBound(
             &plugin, nullptr, cocclCompressorOperationCompress,
             1024, 4, ncclFloat32, &bytes) == ncclSuccess);
  EXPECT(bytes == 4096);

  plugin.getEncodedSizeBound = estimate;
  mode = EstimatorUnavailable;
  EXPECT(cocclQueryCompressorEncodedSizeBound(
             &plugin, nullptr, cocclCompressorOperationCompress,
             1024, 4, ncclFloat32, &bytes) == ncclSuccess);
  EXPECT(bytes == 4096);
}

void checkDispatch() {
  cocclCompressorPlugin plugin = {};
  plugin.getEncodedSizeBound = estimate;
  int config = 7;
  expectedConfig = &config;
  mode = EstimatorExact;
  size_t bytes = 0;
  EXPECT(cocclQueryCompressorEncodedSizeBound(
             &plugin, expectedConfig,
             cocclCompressorOperationDecompressReduceCompress,
             1024, 4, ncclFloat16, &bytes) == ncclSuccess);
  EXPECT(bytes == 1024);
  EXPECT(observed.structSize == sizeof(cocclCompressorSizeQuery));
  EXPECT(observed.operation ==
         cocclCompressorOperationDecompressReduceCompress);
  EXPECT(observed.elements == 1024 && observed.chunks == 4);
  EXPECT(observed.datatype == ncclFloat16);
  EXPECT(observed.config == expectedConfig);
}

void checkErrors() {
  cocclCompressorPlugin plugin = {};
  plugin.getEncodedSizeBound = estimate;
  size_t bytes = 0;

  mode = EstimatorZero;
  EXPECT(cocclQueryCompressorEncodedSizeBound(
             &plugin, nullptr, cocclCompressorOperationCompress,
             1024, 4, ncclFloat32, &bytes) == ncclInvalidArgument);
  mode = EstimatorUnaligned;
  EXPECT(cocclQueryCompressorEncodedSizeBound(
             &plugin, nullptr, cocclCompressorOperationCompress,
             1024, 4, ncclFloat32, &bytes) == ncclInvalidUsage);
  mode = EstimatorError;
  EXPECT(cocclQueryCompressorEncodedSizeBound(
             &plugin, nullptr, cocclCompressorOperationCompress,
             1024, 4, ncclFloat32, &bytes) == ncclSystemError);

  EXPECT(cocclQueryCompressorEncodedSizeBound(
             &plugin, nullptr, cocclCompressorOperationCompress,
             1025, 4, ncclFloat32, &bytes) == ncclInvalidArgument);
  EXPECT(cocclQueryCompressorEncodedSizeBound(
             &plugin, nullptr, cocclCompressorOperationCompress,
             SIZE_MAX / 2 + 1, 1, ncclFloat32,
             &bytes) == ncclInvalidArgument);
}

}  // namespace

int main() {
  checkRawFallback();
  checkDispatch();
  checkErrors();
  std::printf("coccl M11 size query: PASS\n");
  return 0;
}
