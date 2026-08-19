#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr size_t kCorrectnessElements = 4096;
int gWorldRank = 0;

enum class Operation {
  AllToAll,
  AllGather,
  ReduceScatter,
  AllReduceOneShot,
  AllReduceTwoShot,
  SendRecv,
};

struct Options {
  std::string mode;
  std::string datatype;
  std::string operation = "allgather";
  std::string path = "compressed";
  size_t elements = 0;
  int warmups = 20;
  int iterations = 30;
  int depth = 1;
};

void fail(const char* expression, const char* detail,
          const char* file, int line) {
  std::fprintf(stderr, "rank %d: %s failed at %s:%d: %s\n",
               gWorldRank, expression, file, line, detail);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

#define MPICHECK(call)                                                        \
  do {                                                                        \
    const int result = (call);                                                \
    if (result != MPI_SUCCESS) {                                              \
      char error[MPI_MAX_ERROR_STRING];                                       \
      int length = 0;                                                         \
      MPI_Error_string(result, error, &length);                               \
      fail(#call, error, __FILE__, __LINE__);                                 \
    }                                                                         \
  } while (0)

#define CUDACHECK(call)                                                       \
  do {                                                                        \
    const cudaError_t result = (call);                                        \
    if (result != cudaSuccess) {                                              \
      fail(#call, cudaGetErrorString(result), __FILE__, __LINE__);            \
    }                                                                         \
  } while (0)

#define NCCLCHECK(call)                                                       \
  do {                                                                        \
    const ncclResult_t result = (call);                                       \
    if (result != ncclSuccess) {                                              \
      fail(#call, ncclGetErrorString(result), __FILE__, __LINE__);            \
    }                                                                         \
  } while (0)

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index + 1 < argc; index += 2) {
    const std::string key = argv[index];
    const std::string value = argv[index + 1];
    if (key == "--mode") options.mode = value;
    else if (key == "--datatype") options.datatype = value;
    else if (key == "--operation") options.operation = value;
    else if (key == "--path") options.path = value;
    else if (key == "--elements") {
      options.elements = std::strtoull(value.c_str(), nullptr, 10);
    } else if (key == "--warmups") {
      options.warmups = std::atoi(value.c_str());
    } else if (key == "--iterations") {
      options.iterations = std::atoi(value.c_str());
    } else if (key == "--depth") {
      options.depth = std::atoi(value.c_str());
    } else {
      fail("command line", "unknown option", __FILE__, __LINE__);
    }
  }
  if (options.mode.empty() || options.datatype.empty()) {
    fail("command line", "missing mode or datatype", __FILE__, __LINE__);
  }
  return options;
}

const char* operationName(Operation operation) {
  switch (operation) {
    case Operation::AllToAll: return "alltoall";
    case Operation::AllGather: return "allgather";
    case Operation::ReduceScatter: return "reducescatter-oneshot";
    case Operation::AllReduceOneShot: return "allreduce-oneshot";
    case Operation::AllReduceTwoShot: return "allreduce-twoshot";
    case Operation::SendRecv: return "sendrecv";
  }
  return "unknown";
}

Operation parseOperation(const std::string& value) {
  if (value == "alltoall") return Operation::AllToAll;
  if (value == "allgather") return Operation::AllGather;
  if (value == "reducescatter") return Operation::ReduceScatter;
  if (value == "allreduce") return Operation::AllReduceOneShot;
  if (value == "sendrecv") return Operation::SendRecv;
  fail("operation", "unknown operation", __FILE__, __LINE__);
  return Operation::AllGather;
}

size_t outputElements(Operation operation, size_t elements, int ranks) {
  if (operation == Operation::AllGather) return elements * ranks;
  if (operation == Operation::ReduceScatter) return elements / ranks;
  return elements;
}

template <typename T>
std::vector<T> makeInput(int rank, size_t elements, bool compressible) {
  std::vector<T> values(elements);
  for (size_t index = 0; index < elements; ++index) {
    const int64_t value = compressible
        ? 0 : (static_cast<int64_t>(rank) * 23 +
               static_cast<int64_t>(index % 101) * 7) % 101 - 50;
    values[index] = static_cast<T>(value);
  }
  return values;
}

void runPublic(Operation operation, const void* input, void* output,
               size_t elements, ncclDataType_t datatype, ncclComm_t comm,
               cudaStream_t stream, int rank, int ranks) {
  switch (operation) {
    case Operation::AllToAll:
      NCCLCHECK(ncclAllToAll(input, output, elements / ranks, datatype,
                             comm, stream));
      return;
    case Operation::AllGather:
      NCCLCHECK(ncclAllGather(input, output, elements, datatype,
                              comm, stream));
      return;
    case Operation::ReduceScatter:
      NCCLCHECK(ncclReduceScatter(input, output, elements / ranks, datatype,
                                  ncclSum, comm, stream));
      return;
    case Operation::AllReduceOneShot:
    case Operation::AllReduceTwoShot:
      NCCLCHECK(ncclAllReduce(input, output, elements, datatype, ncclSum,
                              comm, stream));
      return;
    case Operation::SendRecv: {
      const int previous = (rank + ranks - 1) % ranks;
      const int next = (rank + 1) % ranks;
      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(ncclRecv(output, elements, datatype, previous, comm, stream));
      NCCLCHECK(ncclSend(input, elements, datatype, next, comm, stream));
      NCCLCHECK(ncclGroupEnd());
      return;
    }
  }
}

void runCompressed(Operation operation, const void* input, void* output,
                   size_t elements, ncclDataType_t datatype, ncclComm_t comm,
                   cudaStream_t stream, int rank, int ranks) {
  switch (operation) {
    case Operation::AllToAll:
      NCCLCHECK(cocclAllToAllComp(input, output, elements / ranks, datatype,
                                  comm, stream));
      return;
    case Operation::AllGather:
      NCCLCHECK(cocclAllGatherComp(input, output, elements, datatype,
                                   comm, stream));
      return;
    case Operation::ReduceScatter:
      NCCLCHECK(cocclReduceScatterCompOneShot(
          input, output, elements / ranks, datatype, ncclSum, comm, stream));
      return;
    case Operation::AllReduceOneShot:
      NCCLCHECK(cocclAllReduceCompOneShot(
          input, output, elements, datatype, ncclSum, comm, stream));
      return;
    case Operation::AllReduceTwoShot:
      NCCLCHECK(cocclAllReduceCompTwoShot(
          input, output, elements, datatype, ncclSum, comm, stream));
      return;
    case Operation::SendRecv: {
      const int previous = (rank + ranks - 1) % ranks;
      const int next = (rank + 1) % ranks;
      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(cocclRecvDecomp(output, elements, datatype, previous,
                                comm, stream));
      NCCLCHECK(cocclSendComp(input, elements, datatype, next, comm, stream));
      NCCLCHECK(ncclGroupEnd());
      return;
    }
  }
}

template <typename T>
void runCorrectnessCase(Operation operation, ncclDataType_t datatype,
                        const Options& options, ncclComm_t nativeComm,
                        ncclComm_t compressedComm, cudaStream_t nativeStream,
                        cudaStream_t compressedStream, int rank, int ranks) {
  const size_t inputElements = kCorrectnessElements;
  const size_t resultElements = outputElements(operation, inputElements,
                                               ranks);
  const size_t inputBytes = inputElements * sizeof(T);
  const size_t outputBytes = resultElements * sizeof(T);
  T* input = nullptr;
  T* nativeOutput = nullptr;
  T* compressedOutput = nullptr;
  CUDACHECK(cudaMalloc(&input, inputBytes));
  CUDACHECK(cudaMalloc(&nativeOutput, outputBytes));
  CUDACHECK(cudaMalloc(&compressedOutput, outputBytes));
  const std::vector<T> hostInput = makeInput<T>(rank, inputElements, false);
  CUDACHECK(cudaMemcpy(input, hostInput.data(), inputBytes,
                       cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(nativeOutput, 0, outputBytes));
  CUDACHECK(cudaMemset(compressedOutput, 0, outputBytes));

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  runPublic(operation, input, nativeOutput, inputElements, datatype,
            nativeComm, nativeStream, rank, ranks);
  CUDACHECK(cudaStreamSynchronize(nativeStream));
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  runCompressed(operation, input, compressedOutput, inputElements, datatype,
                compressedComm, compressedStream, rank, ranks);
  CUDACHECK(cudaStreamSynchronize(compressedStream));

  std::vector<T> nativeHost(resultElements);
  std::vector<T> compressedHost(resultElements);
  CUDACHECK(cudaMemcpy(nativeHost.data(), nativeOutput, outputBytes,
                       cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(compressedHost.data(), compressedOutput, outputBytes,
                       cudaMemcpyDeviceToHost));
  unsigned long long localMismatches = 0;
  for (size_t index = 0; index < resultElements; ++index) {
    localMismatches += nativeHost[index] != compressedHost[index];
  }
  unsigned long long mismatches = 0;
  MPICHECK(MPI_Reduce(&localMismatches, &mismatches, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD));
  if (rank == 0) {
    std::printf("COCCL_INTEGER_CORRECTNESS operation=%s datatype=%s "
                "depth=%d elements=%zu mismatches=%llu\n",
                operationName(operation), options.datatype.c_str(),
                options.depth, resultElements * static_cast<size_t>(ranks),
                mismatches);
    std::fflush(stdout);
  }
  if (mismatches != 0) {
    fail("integer correctness", "native and dietGPU output differ",
         __FILE__, __LINE__);
  }

  CUDACHECK(cudaFree(compressedOutput));
  CUDACHECK(cudaFree(nativeOutput));
  CUDACHECK(cudaFree(input));
}

template <typename T>
void runCorrectness(const Options& options, ncclDataType_t datatype,
                    ncclComm_t nativeComm, ncclComm_t compressedComm,
                    cudaStream_t nativeStream, cudaStream_t compressedStream,
                    int rank, int ranks) {
  const Operation operations[] = {
      Operation::AllToAll, Operation::AllGather, Operation::ReduceScatter,
      Operation::AllReduceOneShot, Operation::AllReduceTwoShot,
      Operation::SendRecv};
  for (Operation operation : operations) {
    runCorrectnessCase<T>(operation, datatype, options, nativeComm,
                          compressedComm, nativeStream, compressedStream,
                          rank, ranks);
  }
}

template <typename T>
void runFallback(const Options& options, ncclDataType_t datatype,
                 ncclComm_t comm, cudaStream_t stream, int rank, int ranks) {
  const size_t elements = kCorrectnessElements;
  const size_t outputCount = elements * ranks;
  const size_t inputBytes = elements * sizeof(T);
  const size_t outputBytes = outputCount * sizeof(T);
  T* input = nullptr;
  T* output = nullptr;
  CUDACHECK(cudaMalloc(&input, inputBytes));
  CUDACHECK(cudaMalloc(&output, outputBytes));
  const std::vector<T> hostInput = makeInput<T>(rank, elements, false);
  CUDACHECK(cudaMemcpy(input, hostInput.data(), inputBytes,
                       cudaMemcpyHostToDevice));
  NCCLCHECK(ncclAllGather(input, output, elements, datatype, comm, stream));
  CUDACHECK(cudaStreamSynchronize(stream));
  std::vector<T> hostOutput(outputCount);
  CUDACHECK(cudaMemcpy(hostOutput.data(), output, outputBytes,
                       cudaMemcpyDeviceToHost));
  unsigned long long localMismatches = 0;
  for (int peer = 0; peer < ranks; ++peer) {
    const std::vector<T> expected = makeInput<T>(peer, elements, false);
    for (size_t index = 0; index < elements; ++index) {
      localMismatches +=
          hostOutput[static_cast<size_t>(peer) * elements + index] !=
          expected[index];
    }
  }
  unsigned long long mismatches = 0;
  MPICHECK(MPI_Reduce(&localMismatches, &mismatches, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD));
  if (rank == 0) {
    std::printf("COCCL_INTEGER_FALLBACK datatype=%s mismatches=%llu\n",
                options.datatype.c_str(), mismatches);
    std::fflush(stdout);
  }
  if (mismatches != 0) {
    fail("integer fallback", "native AllGather result is incorrect",
         __FILE__, __LINE__);
  }
  CUDACHECK(cudaFree(output));
  CUDACHECK(cudaFree(input));
}

template <typename T>
void runPerformance(const Options& options, ncclDataType_t datatype,
                    ncclComm_t comm, cudaStream_t stream,
                    int rank, int ranks) {
  const Operation operation = parseOperation(options.operation);
  const size_t elements = options.elements;
  const size_t resultElements = outputElements(operation, elements, ranks);
  const size_t inputBytes = elements * sizeof(T);
  const size_t outputBytes = resultElements * sizeof(T);
  T* input = nullptr;
  T* output = nullptr;
  CUDACHECK(cudaMalloc(&input, inputBytes));
  CUDACHECK(cudaMalloc(&output, outputBytes));
  CUDACHECK(cudaMemset(input, 0, inputBytes));
  CUDACHECK(cudaMemset(output, 0, outputBytes));

  for (int iteration = 0; iteration < options.warmups; ++iteration) {
    runPublic(operation, input, output, elements, datatype, comm, stream,
              rank, ranks);
  }
  CUDACHECK(cudaStreamSynchronize(stream));
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  cudaEvent_t begin;
  cudaEvent_t end;
  CUDACHECK(cudaEventCreate(&begin));
  CUDACHECK(cudaEventCreate(&end));
  CUDACHECK(cudaEventRecord(begin, stream));
  for (int iteration = 0; iteration < options.iterations; ++iteration) {
    runPublic(operation, input, output, elements, datatype, comm, stream,
              rank, ranks);
  }
  CUDACHECK(cudaEventRecord(end, stream));
  CUDACHECK(cudaEventSynchronize(end));
  float elapsedMs = 0.0f;
  CUDACHECK(cudaEventElapsedTime(&elapsedMs, begin, end));
  const double localUs = elapsedMs * 1000.0 / options.iterations;
  double maxUs = 0.0;
  MPICHECK(MPI_Reduce(&localUs, &maxUs, 1, MPI_DOUBLE, MPI_MAX, 0,
                      MPI_COMM_WORLD));
  if (rank == 0) {
    const double bandwidth = static_cast<double>(inputBytes) / maxUs / 1.0e3;
    std::printf("COCCL_INTEGER_PERF operation=%s datatype=%s path=%s "
                "input_bytes=%zu warmups=%d iterations=%d "
                "latency_us=%.3f input_GBps=%.3f\n",
                options.operation.c_str(), options.datatype.c_str(),
                options.path.c_str(), inputBytes, options.warmups,
                options.iterations, maxUs, bandwidth);
    std::fflush(stdout);
  }
  CUDACHECK(cudaEventDestroy(end));
  CUDACHECK(cudaEventDestroy(begin));
  CUDACHECK(cudaFree(output));
  CUDACHECK(cudaFree(input));
}

template <typename T>
void dispatch(const Options& options, ncclDataType_t datatype,
              ncclComm_t nativeComm, ncclComm_t compressedComm,
              cudaStream_t nativeStream, cudaStream_t compressedStream,
              int rank, int ranks) {
  if (options.mode == "correctness") {
    runCorrectness<T>(options, datatype, nativeComm, compressedComm,
                      nativeStream, compressedStream, rank, ranks);
  } else if (options.mode == "fallback") {
    runFallback<T>(options, datatype, compressedComm, compressedStream,
                   rank, ranks);
  } else if (options.mode == "performance") {
    runPerformance<T>(options, datatype, compressedComm, compressedStream,
                      rank, ranks);
  } else {
    fail("mode", "unknown mode", __FILE__, __LINE__);
  }
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int ranks = 0;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  gWorldRank = rank;
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
  const Options options = parseOptions(argc, argv);

  MPI_Comm localComm;
  MPICHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                               MPI_INFO_NULL, &localComm));
  int localRank = 0;
  MPICHECK(MPI_Comm_rank(localComm, &localRank));
  CUDACHECK(cudaSetDevice(localRank));

  ncclUniqueId nativeId;
  ncclUniqueId compressedId;
  if (rank == 0) {
    NCCLCHECK(ncclGetUniqueId(&nativeId));
    NCCLCHECK(ncclGetUniqueId(&compressedId));
  }
  MPICHECK(MPI_Bcast(&nativeId, sizeof(nativeId), MPI_BYTE, 0,
                     MPI_COMM_WORLD));
  MPICHECK(MPI_Bcast(&compressedId, sizeof(compressedId), MPI_BYTE, 0,
                     MPI_COMM_WORLD));
  ncclComm_t nativeComm = nullptr;
  ncclComm_t compressedComm = nullptr;
  NCCLCHECK(ncclCommInitRank(&nativeComm, ranks, nativeId, rank));
  NCCLCHECK(ncclCommInitRank(&compressedComm, ranks, compressedId, rank));
  cudaStream_t nativeStream = nullptr;
  cudaStream_t compressedStream = nullptr;
  CUDACHECK(cudaStreamCreateWithFlags(&nativeStream, cudaStreamNonBlocking));
  CUDACHECK(cudaStreamCreateWithFlags(&compressedStream,
                                       cudaStreamNonBlocking));

  if (options.datatype == "int8") {
    dispatch<int8_t>(options, ncclInt8, nativeComm, compressedComm,
                     nativeStream, compressedStream, rank, ranks);
  } else if (options.datatype == "int32") {
    dispatch<int32_t>(options, ncclInt32, nativeComm, compressedComm,
                      nativeStream, compressedStream, rank, ranks);
  } else if (options.datatype == "int64") {
    dispatch<int64_t>(options, ncclInt64, nativeComm, compressedComm,
                      nativeStream, compressedStream, rank, ranks);
  } else {
    fail("datatype", "unknown datatype", __FILE__, __LINE__);
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
