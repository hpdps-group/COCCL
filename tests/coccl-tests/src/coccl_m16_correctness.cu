#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr size_t kElementsPerRank = 4096;
int gWorldRank = 0;

enum class Operation {
  AllToAll,
  AllGather,
  ReduceScatterOneShot,
  ReduceScatterTwoShot,
  AllReduceOneShot,
  AllReduceTwoShot,
  AllReduceTripleShot,
  SendRecv,
};

struct Options {
  std::string suite;
  std::string compressor;
  std::string datatype;
  std::string topology;
  int depth = 1;
};

void fail(const char* expression, const char* detail, int rank,
          const char* file, int line) {
  std::fprintf(stderr, "rank %d: %s failed at %s:%d: %s\n", rank,
               expression, file, line, detail);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

#define MPICHECK(call)                                                        \
  do {                                                                        \
    const int result = (call);                                                \
    if (result != MPI_SUCCESS) {                                              \
      char error[MPI_MAX_ERROR_STRING];                                       \
      int length = 0;                                                         \
      MPI_Error_string(result, error, &length);                               \
      fail(#call, error, gWorldRank, __FILE__, __LINE__);                     \
    }                                                                         \
  } while (0)

#define CUDACHECK(call)                                                       \
  do {                                                                        \
    const cudaError_t result = (call);                                        \
    if (result != cudaSuccess) {                                              \
      fail(#call, cudaGetErrorString(result), gWorldRank, __FILE__, __LINE__);\
    }                                                                         \
  } while (0)

#define NCCLCHECK(call)                                                       \
  do {                                                                        \
    const ncclResult_t result = (call);                                       \
    if (result != ncclSuccess) {                                              \
      fail(#call, ncclGetErrorString(result), gWorldRank, __FILE__, __LINE__);\
    }                                                                         \
  } while (0)

Options parseOptions(int argc, char** argv, int worldRank) {
  Options options;
  for (int index = 1; index + 1 < argc; index += 2) {
    const std::string key = argv[index];
    const std::string value = argv[index + 1];
    if (key == "--suite") options.suite = value;
    else if (key == "--compressor") options.compressor = value;
    else if (key == "--datatype") options.datatype = value;
    else if (key == "--topology") options.topology = value;
    else if (key == "--depth") options.depth = std::atoi(value.c_str());
    else fail("command line", "unknown option", worldRank, __FILE__, __LINE__);
  }
  if (options.suite.empty() || options.compressor.empty() ||
      options.datatype.empty() || options.topology.empty()) {
    fail("command line", "missing required option", worldRank, __FILE__,
         __LINE__);
  }
  return options;
}

const char* operationName(Operation operation) {
  switch (operation) {
    case Operation::AllToAll: return "alltoall";
    case Operation::AllGather: return "allgather";
    case Operation::ReduceScatterOneShot: return "reducescatter";
    case Operation::ReduceScatterTwoShot: return "reducescatter";
    case Operation::AllReduceOneShot: return "allreduce";
    case Operation::AllReduceTwoShot: return "allreduce";
    case Operation::AllReduceTripleShot: return "allreduce";
    case Operation::SendRecv: return "sendrecv";
  }
  return "unknown";
}

const char* algorithmName(Operation operation, bool subAdd) {
  if (subAdd) return "subadd";
  switch (operation) {
    case Operation::ReduceScatterOneShot:
    case Operation::AllReduceOneShot: return "oneshot";
    case Operation::ReduceScatterTwoShot:
    case Operation::AllReduceTwoShot: return "twoshot";
    case Operation::AllReduceTripleShot: return "tripleshot";
    default: return "default";
  }
}

size_t outputElements(Operation operation, int ranks) {
  switch (operation) {
    case Operation::AllGather: return kElementsPerRank * ranks;
    case Operation::ReduceScatterOneShot:
    case Operation::ReduceScatterTwoShot: return kElementsPerRank / ranks;
    default: return kElementsPerRank;
  }
}

template <typename T>
T fromFloat(float value);

template <>
float fromFloat<float>(float value) {
  return value;
}

template <>
__half fromFloat<__half>(float value) {
  return __float2half(value);
}

template <>
__nv_bfloat16 fromFloat<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <typename T>
float toFloat(T value);

template <>
float toFloat<float>(float value) {
  return value;
}

template <>
float toFloat<__half>(__half value) {
  return __half2float(value);
}

template <>
float toFloat<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
std::vector<T> makeInput(int rank, int phase) {
  std::vector<T> values(kElementsPerRank);
  const size_t phaseOffset = static_cast<size_t>(phase) * kElementsPerRank;
  for (size_t index = 0; index < values.size(); ++index) {
    float value = static_cast<float>(
        static_cast<size_t>(rank) * kElementsPerRank + phaseOffset + index);
    if (std::is_same<T, __half>::value) value /= 10.0f;
    values[index] = fromFloat<T>(value);
  }
  return values;
}

void runNative(Operation operation, const void* input, void* output,
               ncclDataType_t datatype, ncclComm_t comm,
               cudaStream_t stream, int rank, int ranks) {
  switch (operation) {
    case Operation::AllToAll:
      NCCLCHECK(ncclAllToAll(input, output, kElementsPerRank / ranks,
                             datatype, comm, stream));
      return;
    case Operation::AllGather:
      NCCLCHECK(ncclAllGather(input, output, kElementsPerRank, datatype,
                              comm, stream));
      return;
    case Operation::ReduceScatterOneShot:
    case Operation::ReduceScatterTwoShot:
      NCCLCHECK(ncclReduceScatter(input, output, kElementsPerRank / ranks,
                                  datatype, ncclSum, comm, stream));
      return;
    case Operation::AllReduceOneShot:
    case Operation::AllReduceTwoShot:
    case Operation::AllReduceTripleShot:
      NCCLCHECK(ncclAllReduce(input, output, kElementsPerRank, datatype,
                              ncclSum, comm, stream));
      return;
    case Operation::SendRecv: {
      const int previous = (rank + ranks - 1) % ranks;
      const int next = (rank + 1) % ranks;
      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(ncclRecv(output, kElementsPerRank, datatype, previous, comm,
                         stream));
      NCCLCHECK(ncclSend(input, kElementsPerRank, datatype, next, comm,
                         stream));
      NCCLCHECK(ncclGroupEnd());
      return;
    }
  }
}

void runCompressed(Operation operation, const void* input, void* output,
                   ncclDataType_t datatype, ncclComm_t comm,
                   cudaStream_t stream, int rank, int ranks) {
#ifdef COCCL_M16_LEGACY_API
  switch (operation) {
    case Operation::AllToAll:
      NCCLCHECK(ncclAlltoAllCompOverlap(
          input, output, kElementsPerRank / ranks, datatype, comm, stream));
      return;
    case Operation::AllGather:
      NCCLCHECK(ncclAllGatherCompOverlap(
          input, output, kElementsPerRank, datatype, comm, stream));
      return;
    case Operation::ReduceScatterOneShot:
      NCCLCHECK(ncclReduceScatterCompOneShotOverlap(
          input, output, kElementsPerRank / ranks, datatype, ncclSum, comm,
          stream));
      return;
    case Operation::ReduceScatterTwoShot:
      NCCLCHECK(ncclReduceScatterCompTwoShotTLOverlap(
          input, output, kElementsPerRank / ranks, datatype, ncclSum, comm,
          stream));
      return;
    case Operation::AllReduceOneShot:
      NCCLCHECK(ncclAllReduceCompOneShot(
          input, output, kElementsPerRank, datatype, ncclSum, comm, stream));
      return;
    case Operation::AllReduceTwoShot:
      NCCLCHECK(ncclAllReduceCompTwoShotOverlap(
          input, output, kElementsPerRank, datatype, ncclSum, comm, stream));
      return;
    case Operation::AllReduceTripleShot:
      NCCLCHECK(ncclAllReduceCompTripleShotTLOverlap(
          input, output, kElementsPerRank, datatype, ncclSum, comm, stream));
      return;
    case Operation::SendRecv: {
      const int previous = (rank + ranks - 1) % ranks;
      const int next = (rank + 1) % ranks;
      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(ncclRecvDecomp(output, kElementsPerRank, datatype, previous,
                               comm, stream));
      NCCLCHECK(ncclSendComp(input, kElementsPerRank, datatype, next, comm,
                             stream));
      NCCLCHECK(ncclGroupEnd());
      return;
    }
  }
#else
  switch (operation) {
    case Operation::AllToAll:
      NCCLCHECK(cocclAllToAllComp(input, output, kElementsPerRank / ranks,
                                  datatype, comm, stream));
      return;
    case Operation::AllGather:
      NCCLCHECK(cocclAllGatherComp(input, output, kElementsPerRank, datatype,
                                   comm, stream));
      return;
    case Operation::ReduceScatterOneShot:
      NCCLCHECK(cocclReduceScatterCompOneShot(
          input, output, kElementsPerRank / ranks, datatype, ncclSum, comm,
          stream));
      return;
    case Operation::ReduceScatterTwoShot:
      NCCLCHECK(cocclReduceScatterCompTwoShot(
          input, output, kElementsPerRank / ranks, datatype, ncclSum, comm,
          stream));
      return;
    case Operation::AllReduceOneShot:
      NCCLCHECK(cocclAllReduceCompOneShot(
          input, output, kElementsPerRank, datatype, ncclSum, comm, stream));
      return;
    case Operation::AllReduceTwoShot:
      NCCLCHECK(cocclAllReduceCompTwoShot(
          input, output, kElementsPerRank, datatype, ncclSum, comm, stream));
      return;
    case Operation::AllReduceTripleShot:
      NCCLCHECK(cocclAllReduceCompTripleShot(
          input, output, kElementsPerRank, datatype, ncclSum, comm, stream));
      return;
    case Operation::SendRecv: {
      const int previous = (rank + ranks - 1) % ranks;
      const int next = (rank + 1) % ranks;
      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(cocclRecvDecomp(output, kElementsPerRank, datatype, previous,
                                comm, stream));
      NCCLCHECK(cocclSendComp(input, kElementsPerRank, datatype, next, comm,
                              stream));
      NCCLCHECK(ncclGroupEnd());
      return;
    }
  }
#endif
}

template <typename T>
void runCase(Operation operation, bool subAdd, ncclDataType_t datatype,
             const Options& options, ncclComm_t nativeComm,
             ncclComm_t compressedComm, cudaStream_t nativeStream,
             cudaStream_t compressedStream, int worldRank, int worldSize) {
  const size_t outputCount = outputElements(operation, worldSize);
  const size_t inputBytes = kElementsPerRank * sizeof(T);
  const size_t outputBytes = outputCount * sizeof(T);
  T* deviceInput = nullptr;
  T* nativeOutput = nullptr;
  T* compressedOutput = nullptr;
  CUDACHECK(cudaMalloc(&deviceInput, inputBytes));
  CUDACHECK(cudaMalloc(&nativeOutput, outputBytes));
  CUDACHECK(cudaMalloc(&compressedOutput, outputBytes));

  if (subAdd) {
    const std::vector<T> initial = makeInput<T>(worldRank, 0);
    CUDACHECK(cudaMemcpy(deviceInput, initial.data(), inputBytes,
                         cudaMemcpyHostToDevice));
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    runCompressed(operation, deviceInput, compressedOutput, datatype,
                  compressedComm, compressedStream, worldRank, worldSize);
    CUDACHECK(cudaStreamSynchronize(compressedStream));
  }

  const std::vector<T> input = makeInput<T>(worldRank, subAdd ? 1 : 0);
  CUDACHECK(cudaMemcpy(deviceInput, input.data(), inputBytes,
                       cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(nativeOutput, 0, outputBytes));
  CUDACHECK(cudaMemset(compressedOutput, 0, outputBytes));

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  runNative(operation, deviceInput, nativeOutput, datatype, nativeComm,
            nativeStream, worldRank, worldSize);
  CUDACHECK(cudaStreamSynchronize(nativeStream));
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  runCompressed(operation, deviceInput, compressedOutput, datatype,
                compressedComm, compressedStream, worldRank, worldSize);
  CUDACHECK(cudaStreamSynchronize(compressedStream));

  std::vector<T> nativeHost(outputCount);
  std::vector<T> compressedHost(outputCount);
  CUDACHECK(cudaMemcpy(nativeHost.data(), nativeOutput, outputBytes,
                       cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(compressedHost.data(), compressedOutput, outputBytes,
                       cudaMemcpyDeviceToHost));

  double localError = 0.0;
  for (size_t index = 0; index < outputCount; ++index) {
    const double expected = static_cast<double>(toFloat(nativeHost[index]));
    const double actual = static_cast<double>(toFloat(compressedHost[index]));
    localError += std::fabs(actual - expected) /
        (std::fabs(expected) + 1.0e-6);
  }
  double globalError = 0.0;
  MPICHECK(MPI_Reduce(&localError, &globalError, 1, MPI_DOUBLE, MPI_SUM, 0,
                      MPI_COMM_WORLD));
  if (worldRank == 0) {
    const double mean = globalError /
        static_cast<double>(outputCount * static_cast<size_t>(worldSize));
    std::printf(
        "COCCL_CORRECTNESS topology=%s rank_count=%d operation=%s "
        "algorithm=%s compressor=%s dtype=%s depth=%d "
        "output_elements=%zu mean_relative_error=%.12e\n",
        options.topology.c_str(), worldSize, operationName(operation),
        algorithmName(operation, subAdd), options.compressor.c_str(),
        options.datatype.c_str(), options.depth, outputCount, mean);
    std::fflush(stdout);
  }

  CUDACHECK(cudaFree(compressedOutput));
  CUDACHECK(cudaFree(nativeOutput));
  CUDACHECK(cudaFree(deviceInput));
}

template <typename T>
void runSuite(const Options& options, ncclDataType_t datatype,
              ncclComm_t nativeComm, ncclComm_t compressedComm,
              cudaStream_t nativeStream, cudaStream_t compressedStream,
              int worldRank, int worldSize) {
  std::vector<Operation> operations;
  bool subAdd = false;
  if (options.suite == "single") {
    operations = {Operation::AllToAll, Operation::AllGather,
                  Operation::ReduceScatterOneShot,
                  Operation::AllReduceOneShot,
                  Operation::AllReduceTwoShot};
#ifndef COCCL_M16_LEGACY_API
    operations.push_back(Operation::SendRecv);
#endif
  } else if (options.suite == "hierarchical") {
    operations = {Operation::ReduceScatterTwoShot,
                  Operation::AllReduceTripleShot};
  } else if (options.suite == "subadd") {
    operations = {Operation::AllGather};
    subAdd = true;
  } else {
    fail("suite", "unknown suite", worldRank, __FILE__, __LINE__);
  }
  for (Operation operation : operations) {
    runCase<T>(operation, subAdd, datatype, options, nativeComm,
               compressedComm, nativeStream, compressedStream, worldRank,
               worldSize);
  }
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int worldRank = 0;
  int worldSize = 0;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
  gWorldRank = worldRank;
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
  const Options options = parseOptions(argc, argv, worldRank);

  MPI_Comm localComm;
  MPICHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                               MPI_INFO_NULL, &localComm));
  int localRank = 0;
  MPICHECK(MPI_Comm_rank(localComm, &localRank));
  CUDACHECK(cudaSetDevice(localRank));

  ncclUniqueId nativeId;
  ncclUniqueId compressedId;
  if (worldRank == 0) {
    NCCLCHECK(ncclGetUniqueId(&nativeId));
    NCCLCHECK(ncclGetUniqueId(&compressedId));
  }
  MPICHECK(MPI_Bcast(&nativeId, sizeof(nativeId), MPI_BYTE, 0,
                     MPI_COMM_WORLD));
  MPICHECK(MPI_Bcast(&compressedId, sizeof(compressedId), MPI_BYTE, 0,
                     MPI_COMM_WORLD));

  ncclComm_t nativeComm = nullptr;
  ncclComm_t compressedComm = nullptr;
  NCCLCHECK(ncclCommInitRank(&nativeComm, worldSize, nativeId, worldRank));
  NCCLCHECK(ncclCommInitRank(&compressedComm, worldSize, compressedId,
                             worldRank));
  cudaStream_t nativeStream = nullptr;
  cudaStream_t compressedStream = nullptr;
  CUDACHECK(cudaStreamCreateWithFlags(&nativeStream, cudaStreamNonBlocking));
  CUDACHECK(cudaStreamCreateWithFlags(&compressedStream,
                                       cudaStreamNonBlocking));

  if (options.datatype == "float") {
    runSuite<float>(options, ncclFloat32, nativeComm, compressedComm,
                    nativeStream, compressedStream, worldRank, worldSize);
  } else if (options.datatype == "half") {
    runSuite<__half>(options, ncclFloat16, nativeComm, compressedComm,
                     nativeStream, compressedStream, worldRank, worldSize);
  } else if (options.datatype == "bfloat16") {
    runSuite<__nv_bfloat16>(options, ncclBfloat16, nativeComm,
                            compressedComm, nativeStream, compressedStream,
                            worldRank, worldSize);
  } else {
    fail("datatype", "unknown datatype", worldRank, __FILE__, __LINE__);
  }

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  CUDACHECK(cudaStreamDestroy(compressedStream));
  CUDACHECK(cudaStreamDestroy(nativeStream));
  NCCLCHECK(ncclCommDestroy(compressedComm));
  NCCLCHECK(ncclCommDestroy(nativeComm));
  MPICHECK(MPI_Comm_free(&localComm));
  MPI_Finalize();
  return 0;
}
