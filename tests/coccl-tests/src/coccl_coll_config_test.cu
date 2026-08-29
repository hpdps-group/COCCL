#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int worldRank = 0;

void fail(const char* call, const char* detail, int line) {
  std::fprintf(stderr, "rank %d line %d: %s failed: %s\n",
               worldRank, line, call, detail);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

#define MPICHECK(call) do { \
  const int result = (call); \
  if (result != MPI_SUCCESS) { \
    char error[MPI_MAX_ERROR_STRING]; \
    int length = 0; \
    MPI_Error_string(result, error, &length); \
    fail(#call, error, __LINE__); \
  } \
} while (0)

#define CUDACHECK(call) do { \
  const cudaError_t result = (call); \
  if (result != cudaSuccess) fail(#call, cudaGetErrorString(result), __LINE__); \
} while (0)

#define NCCLCHECK(call) do { \
  const ncclResult_t result = (call); \
  if (result != ncclSuccess) fail(#call, ncclGetErrorString(result), __LINE__); \
} while (0)

struct Options {
  size_t elements = 8ULL << 20;
  int iterations = 10;
  bool tagOnly = false;
};

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    if (std::strcmp(argv[index], "--tag-only") == 0) {
      options.tagOnly = true;
    } else if (std::strcmp(argv[index], "--elements") == 0) {
      options.elements = std::strtoull(argv[++index], nullptr, 10);
    } else if (std::strcmp(argv[index], "--iterations") == 0) {
      options.iterations = std::atoi(argv[++index]);
    } else {
      fail("command line", "unknown option", __LINE__);
    }
  }
  return options;
}

template <typename Call>
double timeCollective(const char* operation, const char* variant,
                      size_t algorithmBytes, int iterations,
                      cudaStream_t stream, Call call) {
  std::printf("CONFIG_CASE rank=%d operation=%s variant=%s\n",
              worldRank, operation, variant);
  std::fflush(stdout);
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  NCCLCHECK(call());
  CUDACHECK(cudaStreamSynchronize(stream));

  cudaEvent_t begin;
  cudaEvent_t end;
  CUDACHECK(cudaEventCreate(&begin));
  CUDACHECK(cudaEventCreate(&end));
  CUDACHECK(cudaEventRecord(begin, stream));
  for (int iteration = 0; iteration < iterations; ++iteration) {
    NCCLCHECK(call());
  }
  CUDACHECK(cudaEventRecord(end, stream));
  CUDACHECK(cudaEventSynchronize(end));
  float localMs = 0.0f;
  CUDACHECK(cudaEventElapsedTime(&localMs, begin, end));
  CUDACHECK(cudaEventDestroy(end));
  CUDACHECK(cudaEventDestroy(begin));

  float maxMs = 0.0f;
  MPICHECK(MPI_Reduce(&localMs, &maxMs, 1, MPI_FLOAT, MPI_MAX, 0,
                      MPI_COMM_WORLD));
  const double seconds = static_cast<double>(maxMs) /
      (1000.0 * static_cast<double>(iterations));
  const double algBw = static_cast<double>(algorithmBytes) / seconds / 1.0e9;
  if (worldRank == 0) {
    std::printf("CONFIG_RESULT operation=%s variant=%s algbw_gbs=%.3f\n",
                operation, variant, algBw);
    std::fflush(stdout);
  }
  return algBw;
}

void copyInput(float* device, const std::vector<float>& host,
               cudaStream_t stream) {
  CUDACHECK(cudaMemcpyAsync(device, host.data(),
                            host.size() * sizeof(float),
                            cudaMemcpyHostToDevice, stream));
}

void checkOutput(const char* label, const float* device,
                 const std::vector<float>& expected, cudaStream_t stream) {
  std::vector<float> actual(expected.size());
  CUDACHECK(cudaMemcpyAsync(actual.data(), device,
                            actual.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, stream));
  CUDACHECK(cudaStreamSynchronize(stream));
  if (!std::equal(actual.begin(), actual.end(), expected.begin())) {
    fail(label, "output mismatch", __LINE__);
  }
}

ncclCollConfig_t ctaConfig(int ctas) {
  ncclCollConfig_t config = NCCL_COLLCONFIG_INITIALIZER;
  config.minCTAs = ctas;
  config.maxCTAs = ctas;
  return config;
}

ncclCollConfig_t algorithmConfig(const char* algorithm, bool force) {
  ncclCollConfig_t config = NCCL_COLLCONFIG_INITIALIZER;
  config.algSelection = algorithm;
  config.forceAlgSelection = force ? 1 : 0;
  return config;
}

void runAllGatherMatrix(float* send, float* recv, size_t count,
                        int ranks, ncclComm_t comm, cudaStream_t stream,
                        int iterations) {
  std::vector<float> input(count);
  for (size_t index = 0; index < count; ++index) {
    input[index] = static_cast<float>((size_t)worldRank * count + index);
  }
  std::vector<float> expected(count * (size_t)ranks);
  for (int rank = 0; rank < ranks; ++rank) {
    for (size_t index = 0; index < count; ++index) {
      expected[(size_t)rank * count + index] =
          static_cast<float>((size_t)rank * count + index);
    }
  }
  copyInput(send, input, stream);

  ncclCollConfig_t initial = NCCL_COLLCONFIG_INITIALIZER;
  ncclCollConfig_t cta8 = ctaConfig(8);
  ncclCollConfig_t cta9 = ctaConfig(9);
  ncclCollConfig_t cta10 = ctaConfig(10);
  ncclCollConfig_t ring = algorithmConfig("RING_SIMPLE", true);
  const size_t bytes = count * sizeof(float);

  timeCollective("allgather", "plain-before", bytes, iterations, stream,
      [&] { return ncclAllGather(send, recv, count, ncclFloat32,
                                 comm, stream); });
  checkOutput("allgather plain-before", recv, expected, stream);
  timeCollective("allgather", "config-null", bytes, iterations, stream,
      [&] { return ncclAllGatherConfig(send, recv, count, ncclFloat32,
                                       comm, stream, nullptr); });
  checkOutput("allgather config-null", recv, expected, stream);
  timeCollective("allgather", "initializer", bytes, iterations, stream,
      [&] { return ncclAllGatherConfig(send, recv, count, ncclFloat32,
                                       comm, stream, &initial); });
  checkOutput("allgather initializer", recv, expected, stream);
  for (const auto& variant : std::vector<std::pair<const char*, ncclCollConfig_t*>>{
           {"cta8", &cta8}, {"cta9", &cta9}, {"cta10", &cta10},
           {"ring-simple", &ring}}) {
    timeCollective("allgather", variant.first, bytes, iterations, stream,
        [&] { return ncclAllGatherConfig(send, recv, count, ncclFloat32,
                                         comm, stream, variant.second); });
    checkOutput(variant.first, recv, expected, stream);
  }
  timeCollective("allgather", "plain-after", bytes, iterations, stream,
      [&] { return ncclAllGather(send, recv, count, ncclFloat32,
                                 comm, stream); });
  checkOutput("allgather plain-after", recv, expected, stream);

  ncclCollConfig_t optionalFallback =
      algorithmConfig("TREE_SIMPLE", false);
  timeCollective("allgather", "invalid-alg-fallback", bytes, 1, stream,
      [&] { return ncclAllGatherConfig(send, recv, count, ncclFloat32,
                                       comm, stream, &optionalFallback); });
  checkOutput("allgather invalid-alg-fallback", recv, expected, stream);
}

void runSmokeMatrix(float* send, float* recv, size_t count, int ranks,
                    ncclComm_t comm, cudaStream_t stream) {
  ncclCollConfig_t initial = NCCL_COLLCONFIG_INITIALIZER;
  ncclCollConfig_t cta8 = ctaConfig(8);
  ncclCollConfig_t ring = algorithmConfig("RING_SIMPLE", true);
  const size_t chunkBytes = count * sizeof(float);

  std::vector<float> input(count, static_cast<float>(worldRank + 1));
  std::vector<float> expected(count,
      static_cast<float>(ranks * (ranks + 1) / 2));
  copyInput(send, input, stream);
  for (const auto& variant : std::vector<std::pair<const char*, ncclCollConfig_t*>>{
           {"initializer", &initial}, {"cta8", &cta8},
           {"ring-simple", &ring}}) {
    timeCollective("allreduce", variant.first, chunkBytes, 1, stream,
        [&] { return ncclAllReduceConfig(send, recv, count, ncclFloat32,
                                         ncclSum, comm, stream,
                                         variant.second); });
    checkOutput(variant.first, recv, expected, stream);
  }

  input.assign(count * (size_t)ranks, static_cast<float>(worldRank + 1));
  copyInput(send, input, stream);
  for (const auto& variant : std::vector<std::pair<const char*, ncclCollConfig_t*>>{
           {"initializer", &initial}, {"cta8", &cta8},
           {"ring-simple", &ring}}) {
    timeCollective("reducescatter", variant.first, chunkBytes, 1, stream,
        [&] { return ncclReduceScatterConfig(
            send, recv, count, ncclFloat32, ncclSum, comm, stream,
            variant.second); });
    checkOutput(variant.first, recv, expected, stream);
  }

  input.resize(count * (size_t)ranks);
  expected.resize(count * (size_t)ranks);
  for (int peer = 0; peer < ranks; ++peer) {
    std::fill_n(input.begin() + (size_t)peer * count, count,
                static_cast<float>(worldRank * 100 + peer));
    std::fill_n(expected.begin() + (size_t)peer * count, count,
                static_cast<float>(peer * 100 + worldRank));
  }
  copyInput(send, input, stream);
  for (const auto& variant : std::vector<std::pair<const char*, ncclCollConfig_t*>>{
           {"initializer", &initial}, {"cta8", &cta8}}) {
    timeCollective("alltoall", variant.first,
                   chunkBytes * (size_t)ranks, 1, stream,
        [&] { return ncclAlltoAllConfig(send, recv, count, ncclFloat32,
                                        comm, stream, variant.second); });
    checkOutput(variant.first, recv, expected, stream);
  }
  ncclCollConfig_t alltoallFallback =
      algorithmConfig("RING_SIMPLE", false);
  timeCollective("alltoall", "invalid-alg-fallback",
                 chunkBytes * (size_t)ranks, 1, stream,
      [&] { return ncclAlltoAllConfig(send, recv, count, ncclFloat32,
                                      comm, stream, &alltoallFallback); });
  checkOutput("alltoall invalid-alg-fallback", recv, expected, stream);
}

void runTagOnly(float* send, float* recv, size_t count, ncclComm_t comm,
                cudaStream_t stream) {
  std::vector<float> input(count, static_cast<float>(worldRank + 1));
  copyInput(send, input, stream);
  ncclCollConfig_t tagged = NCCL_COLLCONFIG_INITIALIZER;
  tagged.userProfilerTag = 0x1234;
  timeCollective("allgather", "tag-only", count * sizeof(float), 1,
      stream, [&] { return ncclAllGatherConfig(
          send, recv, count, ncclFloat32, comm, stream, &tagged); });
}

void checkForcedInvalidAlgorithm(float* send, float* recv, size_t count,
                                 ncclComm_t comm, cudaStream_t stream) {
  ncclCollConfig_t invalid = algorithmConfig("TREE_SIMPLE", true);
  const ncclResult_t result = ncclAllGatherConfig(
      send, recv, count, ncclFloat32, comm, stream, &invalid);
  if (result != ncclInvalidArgument) {
    fail("forced invalid algorithm", ncclGetErrorString(result), __LINE__);
  }
  if (worldRank == 0) {
    std::printf("CONFIG_EXPECTED_ERROR operation=allgather result=ncclInvalidArgument\n");
    std::fflush(stdout);
  }
}

}  // namespace

int main(int argc, char** argv) {
  MPICHECK(MPI_Init(&argc, &argv));
  int worldSize = 0;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
  const Options options = parseOptions(argc, argv);

  MPI_Comm localComm;
  int localRank = 0;
  MPICHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                               worldRank, MPI_INFO_NULL, &localComm));
  MPICHECK(MPI_Comm_rank(localComm, &localRank));
  CUDACHECK(cudaSetDevice(localRank));

  ncclUniqueId uniqueId;
  if (worldRank == 0) NCCLCHECK(ncclGetUniqueId(&uniqueId));
  MPICHECK(MPI_Bcast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0,
                     MPI_COMM_WORLD));
  ncclComm_t comm = nullptr;
  NCCLCHECK(ncclCommInitRank(&comm, worldSize, uniqueId, worldRank));
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  const size_t bufferElements = options.elements * (size_t)worldSize;
  float* send = nullptr;
  float* recv = nullptr;
  CUDACHECK(cudaMalloc(&send, bufferElements * sizeof(float)));
  CUDACHECK(cudaMalloc(&recv, bufferElements * sizeof(float)));

  if (options.tagOnly) {
    runTagOnly(send, recv, options.elements, comm, stream);
  } else {
    runAllGatherMatrix(send, recv, options.elements, worldSize, comm,
                       stream, options.iterations);
    runSmokeMatrix(send, recv, options.elements, worldSize, comm, stream);
    checkForcedInvalidAlgorithm(send, recv, options.elements, comm, stream);
  }

  CUDACHECK(cudaFree(recv));
  CUDACHECK(cudaFree(send));
  CUDACHECK(cudaStreamDestroy(stream));
  NCCLCHECK(ncclCommDestroy(comm));
  MPICHECK(MPI_Comm_free(&localComm));
  MPICHECK(MPI_Finalize());
  if (worldRank == 0) std::printf("coccl per-collective config: PASS\n");
  return 0;
}
