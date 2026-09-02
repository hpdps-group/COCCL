#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr size_t kCorrectnessElements = 4096;
int gWorldRank = 0;

__global__ void fillRatio130Kernel(unsigned char* data, size_t bytes,
                                   uint32_t seed) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < bytes; index += (size_t)blockDim.x * gridDim.x) {
    uint32_t value = static_cast<uint32_t>(index) ^
        static_cast<uint32_t>(index >> 32) ^ seed;
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    data[index] = static_cast<unsigned char>(value & 0x3f);
  }
}
__global__ void fillFp32Ratio130Kernel(float* data, size_t elements,
                                       uint32_t seed) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements; index += (size_t)blockDim.x * gridDim.x) {
    uint32_t state = static_cast<uint32_t>(index) ^
        static_cast<uint32_t>(index >> 32) ^ seed;
    state ^= state >> 16;
    state *= 0x7feb352du;
    state ^= state >> 15;
    const uint32_t bits = 0x3f800000u | (state & 0x007fffffu);
    data[index] = __uint_as_float(bits);
  }
}


__global__ void countMismatchKernel(
    const unsigned char* expected, const unsigned char* actual,
    size_t bytes, unsigned long long* mismatches) {
  __shared__ unsigned long long blockCounts[256];
  unsigned long long count = 0;
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < bytes; index += (size_t)blockDim.x * gridDim.x) {
    count += expected[index] != actual[index];
  }
  blockCounts[threadIdx.x] = count;
  __syncthreads();
  for (int width = blockDim.x / 2; width > 0; width /= 2) {
    if (threadIdx.x < width) {
      blockCounts[threadIdx.x] += blockCounts[threadIdx.x + width];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && blockCounts[0] != 0) {
    atomicAdd(mismatches, blockCounts[0]);
  }
}

enum class Operation {
  AllToAll,
  AllGather,
  AllGatherTwoShot,
  ReduceScatterOneShot,
  ReduceScatterTwoShot,
  AllReduceOneShot,
  AllReduceTwoShot,
  AllReduceTripleShot,
  SendRecv,
  SendRecvCrossNode,
};

struct Options {
  std::string mode;
  std::string datatype;
  std::string operation = "all";
  std::string path = "compressed";
  std::string pattern = "compressible";
  size_t elements = 0;
  size_t prewarmElements = 0;
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

int byteKernelBlocks(size_t bytes) {
  return static_cast<int>(std::min<size_t>(65535, (bytes + 255) / 256));
}

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index + 1 < argc; index += 2) {
    const std::string key = argv[index];
    const std::string value = argv[index + 1];
    if (key == "--mode") options.mode = value;
    else if (key == "--datatype") options.datatype = value;
    else if (key == "--operation") options.operation = value;
    else if (key == "--path") options.path = value;
    else if (key == "--pattern") options.pattern = value;
    else if (key == "--elements") {
      options.elements = std::strtoull(value.c_str(), nullptr, 10);
    } else if (key == "--prewarm-elements") {
      options.prewarmElements = std::strtoull(value.c_str(), nullptr, 10);
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
    case Operation::AllGatherTwoShot: return "allgather-twoshot";
    case Operation::ReduceScatterOneShot: return "reducescatter-oneshot";
    case Operation::ReduceScatterTwoShot: return "reducescatter-twoshot";
    case Operation::AllReduceOneShot: return "allreduce-oneshot";
    case Operation::AllReduceTwoShot: return "allreduce-twoshot";
    case Operation::AllReduceTripleShot: return "allreduce-tripleshot";
    case Operation::SendRecv: return "sendrecv";
    case Operation::SendRecvCrossNode: return "sendrecv-cross-node";
  }
  return "unknown";
}

Operation parseOperation(const std::string& value) {
  if (value == "alltoall") return Operation::AllToAll;
  if (value == "allgather") return Operation::AllGather;
  if (value == "allgather-twoshot") return Operation::AllGatherTwoShot;
  if (value == "reducescatter" || value == "reducescatter-oneshot") {
    return Operation::ReduceScatterOneShot;
  }
  if (value == "reducescatter-twoshot") {
    return Operation::ReduceScatterTwoShot;
  }
  if (value == "allreduce" || value == "allreduce-oneshot") {
    return Operation::AllReduceOneShot;
  }
  if (value == "allreduce-twoshot") return Operation::AllReduceTwoShot;
  if (value == "allreduce-tripleshot") {
    return Operation::AllReduceTripleShot;
  }
  if (value == "sendrecv") return Operation::SendRecv;
  if (value == "sendrecv-cross-node") return Operation::SendRecvCrossNode;
  fail("operation", "unknown operation", __FILE__, __LINE__);
  return Operation::AllGather;
}

size_t outputElements(Operation operation, size_t elements, int ranks) {
  if (operation == Operation::AllGather ||
      operation == Operation::AllGatherTwoShot) {
    return elements * ranks;
  }
  if (operation == Operation::ReduceScatterOneShot ||
      operation == Operation::ReduceScatterTwoShot) {
    return elements / ranks;
  }
  return elements;
}

void sendRecvPeers(Operation operation, int rank, int ranks,
                   int* recvPeer, int* sendPeer) {
  if (operation == Operation::SendRecvCrossNode) {
    *recvPeer = (rank + ranks / 2) % ranks;
    *sendPeer = *recvPeer;
    return;
  }
  *recvPeer = (rank + ranks - 1) % ranks;
  *sendPeer = (rank + 1) % ranks;
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

template <typename T>
std::vector<T> makeRandomInput(int rank, size_t elements) {
  std::vector<T> values(elements);
  uint32_t state = 0x9e3779b9u ^ static_cast<uint32_t>(rank + 1);
  auto* bytes = reinterpret_cast<unsigned char*>(values.data());
  for (size_t index = 0; index < elements * sizeof(T); ++index) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    bytes[index] = static_cast<unsigned char>(state);
  }
  return values;
}

template <typename T>
void fillRatio130Device(T* input, size_t elements, int rank) {
  const size_t bytes = elements * sizeof(T);
  fillRatio130Kernel<<<byteKernelBlocks(bytes), 256>>>(
      reinterpret_cast<unsigned char*>(input), bytes,
      0x85ebca6bu ^ static_cast<uint32_t>(rank + 1));
}

template <>
void fillRatio130Device<float>(float* input, size_t elements, int rank) {
  fillFp32Ratio130Kernel<<<byteKernelBlocks(elements * sizeof(float)), 256>>>(
      input, elements, 0x85ebca6bu ^ static_cast<uint32_t>(rank + 1));
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
    case Operation::AllGatherTwoShot:
      NCCLCHECK(ncclAllGather(input, output, elements, datatype,
                              comm, stream));
      return;
    case Operation::ReduceScatterOneShot:
    case Operation::ReduceScatterTwoShot:
      NCCLCHECK(ncclReduceScatter(input, output, elements / ranks, datatype,
                                  ncclSum, comm, stream));
      return;
    case Operation::AllReduceOneShot:
    case Operation::AllReduceTwoShot:
    case Operation::AllReduceTripleShot:
      NCCLCHECK(ncclAllReduce(input, output, elements, datatype, ncclSum,
                              comm, stream));
      return;
    case Operation::SendRecv:
    case Operation::SendRecvCrossNode: {
      int recvPeer = 0;
      int sendPeer = 0;
      sendRecvPeers(operation, rank, ranks, &recvPeer, &sendPeer);
      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(ncclRecv(output, elements, datatype, recvPeer, comm, stream));
      NCCLCHECK(ncclSend(input, elements, datatype, sendPeer, comm, stream));
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
    case Operation::AllGatherTwoShot:
      NCCLCHECK(cocclAllGatherCompTwoShot(
          input, output, elements, datatype, comm, stream));
      return;
    case Operation::ReduceScatterOneShot:
      NCCLCHECK(cocclReduceScatterCompOneShot(
          input, output, elements / ranks, datatype, ncclSum, comm, stream));
      return;
    case Operation::ReduceScatterTwoShot:
      NCCLCHECK(cocclReduceScatterCompTwoShot(
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
    case Operation::AllReduceTripleShot:
      NCCLCHECK(cocclAllReduceCompTripleShot(
          input, output, elements, datatype, ncclSum, comm, stream));
      return;
    case Operation::SendRecv:
    case Operation::SendRecvCrossNode: {
      int recvPeer = 0;
      int sendPeer = 0;
      sendRecvPeers(operation, rank, ranks, &recvPeer, &sendPeer);
      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(cocclRecvDecomp(output, elements, datatype, recvPeer,
                                comm, stream));
      NCCLCHECK(cocclSendComp(input, elements, datatype, sendPeer,
                              comm, stream));
      NCCLCHECK(ncclGroupEnd());
      return;
    }
  }
}

template <typename T>
void prewarmCompressed(
    Operation operation, ncclDataType_t datatype, const Options& options,
    ncclComm_t comm, cudaStream_t stream, int rank, int ranks) {
  if (options.prewarmElements == 0) return;
  const size_t inputElements = options.prewarmElements;
  const size_t resultElements = outputElements(
      operation, inputElements, ranks);
  T* input = nullptr;
  T* output = nullptr;
  CUDACHECK(cudaMalloc(&input, inputElements * sizeof(T)));
  CUDACHECK(cudaMalloc(&output, resultElements * sizeof(T)));
  CUDACHECK(cudaMemset(input, 0, inputElements * sizeof(T)));
  CUDACHECK(cudaMemset(output, 0, resultElements * sizeof(T)));
  runCompressed(operation, input, output, inputElements, datatype,
                comm, stream, rank, ranks);
  CUDACHECK(cudaStreamSynchronize(stream));
  CUDACHECK(cudaFree(output));
  CUDACHECK(cudaFree(input));
}

template <typename T>
void runCorrectnessCase(Operation operation, ncclDataType_t datatype,
                        const Options& options, ncclComm_t nativeComm,
                        ncclComm_t compressedComm, cudaStream_t nativeStream,
                        cudaStream_t compressedStream, int rank, int ranks) {
  prewarmCompressed<T>(operation, datatype, options, compressedComm,
                       compressedStream, rank, ranks);
  const size_t inputElements = options.elements == 0
      ? kCorrectnessElements : options.elements;
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
  if (options.pattern == "ratio130") {
    fillRatio130Device(input, inputElements, rank);
    CUDACHECK(cudaGetLastError());
  } else {
    const std::vector<T> hostInput = options.pattern == "mixed"
        ? (rank % 2 == 0
               ? makeInput<T>(rank, inputElements, true)
               : makeRandomInput<T>(rank, inputElements))
        : (options.pattern == "random"
               ? makeRandomInput<T>(rank, inputElements)
               : makeInput<T>(rank, inputElements, false));
    CUDACHECK(cudaMemcpy(input, hostInput.data(), inputBytes,
                         cudaMemcpyHostToDevice));
  }
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

  unsigned long long* deviceMismatches = nullptr;
  CUDACHECK(cudaMalloc(&deviceMismatches, sizeof(*deviceMismatches)));
  CUDACHECK(cudaMemset(deviceMismatches, 0, sizeof(*deviceMismatches)));
  countMismatchKernel<<<byteKernelBlocks(outputBytes), 256>>>(
      reinterpret_cast<const unsigned char*>(nativeOutput),
      reinterpret_cast<const unsigned char*>(compressedOutput), outputBytes,
      deviceMismatches);
  CUDACHECK(cudaGetLastError());
  unsigned long long localMismatches = 0;
  CUDACHECK(cudaMemcpy(&localMismatches, deviceMismatches,
                       sizeof(localMismatches), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaFree(deviceMismatches));
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
  std::vector<Operation> operations;
  if (options.operation == "all") {
    operations = {
        Operation::AllReduceOneShot,
        Operation::AllToAll, Operation::AllGather,
        Operation::AllGatherTwoShot,
        Operation::ReduceScatterOneShot, Operation::ReduceScatterTwoShot,
        Operation::AllReduceTwoShot,
        Operation::AllReduceTripleShot, Operation::SendRecv,
        Operation::SendRecvCrossNode};
  } else {
    operations.push_back(parseOperation(options.operation));
  }
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
  prewarmCompressed<T>(operation, datatype, options, comm, stream,
                       rank, ranks);
  const size_t elements = options.elements;
  const size_t resultElements = outputElements(operation, elements, ranks);
  const size_t inputBytes = elements * sizeof(T);
  const size_t outputBytes = resultElements * sizeof(T);
  T* input = nullptr;
  T* output = nullptr;
  CUDACHECK(cudaMalloc(&input, inputBytes));
  CUDACHECK(cudaMalloc(&output, outputBytes));
  const bool random = options.pattern == "random" ||
      (options.pattern == "mixed" && rank % 2 != 0);
  if (options.pattern == "compressible" ||
      (options.pattern == "mixed" && !random)) {
    CUDACHECK(cudaMemset(input, 0, inputBytes));
  } else if (random) {
    const std::vector<T> hostInput = makeRandomInput<T>(rank, elements);
    CUDACHECK(cudaMemcpy(input, hostInput.data(), inputBytes,
                         cudaMemcpyHostToDevice));
  } else if (options.pattern == "ratio130") {
    fillRatio130Device(input, elements, rank);
    CUDACHECK(cudaGetLastError());
  } else {
    fail("pattern", "unknown input pattern", __FILE__, __LINE__);
  }
  CUDACHECK(cudaMemset(output, 0, outputBytes));

  const auto execute = options.path == "compressed" ? runCompressed
                                                      : runPublic;

  for (int iteration = 0; iteration < options.warmups; ++iteration) {
    execute(operation, input, output, elements, datatype, comm, stream,
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
    execute(operation, input, output, elements, datatype, comm, stream,
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
                "pattern=%s depth=%d input_bytes=%zu warmups=%d iterations=%d "
                "latency_us=%.3f input_GBps=%.3f\n",
                options.operation.c_str(), options.datatype.c_str(),
                options.path.c_str(), options.pattern.c_str(), options.depth,
                inputBytes, options.warmups, options.iterations, maxUs,
                bandwidth);
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
  } else if (options.datatype == "fp16") {
    dispatch<__half>(options, ncclFloat16, nativeComm, compressedComm,
                     nativeStream, compressedStream, rank, ranks);
  } else if (options.datatype == "bf16") {
    dispatch<__nv_bfloat16>(options, ncclBfloat16, nativeComm,
                            compressedComm, nativeStream, compressedStream,
                            rank, ranks);
  } else if (options.datatype == "fp32") {
    dispatch<float>(options, ncclFloat32, nativeComm, compressedComm,
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
