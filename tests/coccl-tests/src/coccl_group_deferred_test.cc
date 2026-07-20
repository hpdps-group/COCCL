#include <cuda_runtime.h>
#include <nccl.h>

#include <pthread.h>
#include <stdio.h>
#include <thread>
#include <vector>

#define TEST_CUDA(call) do {                                                   \
  cudaError_t result = (call);                                                 \
  if (result != cudaSuccess) {                                                 \
    fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(result));    \
    return 1;                                                                  \
  }                                                                            \
} while (0)

#define TEST_NCCL(call) do {                                                   \
  ncclResult_t result = (call);                                                \
  if (result != ncclSuccess) {                                                 \
    fprintf(stderr, "%s failed: %s\n", #call, ncclGetErrorString(result));    \
    return 1;                                                                  \
  }                                                                            \
} while (0)

static int runRank(int rank, ncclComm_t comm, pthread_barrier_t* barrier) {
  constexpr size_t count = 256 * 1024;
  constexpr size_t bytes = count * sizeof(float);
  constexpr size_t outputBytes = 2 * bytes;

  cudaStream_t stream = nullptr;
  float* input = nullptr;
  float* output = nullptr;

  TEST_CUDA(cudaSetDevice(rank));
  TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  TEST_CUDA(cudaMalloc(&input, bytes));
  TEST_CUDA(cudaMalloc(&output, outputBytes));

  std::vector<float> hostInput(count, 1.0f);
  TEST_CUDA(cudaMemcpy(input, hostInput.data(), bytes, cudaMemcpyHostToDevice));
  TEST_CUDA(cudaMemset(output, 0, outputBytes));
  TEST_CUDA(cudaStreamSynchronize(stream));
  pthread_barrier_wait(barrier);

  // Inner GroupEnd must only reduce nesting depth. No COCCL kernel may be
  // submitted until the outermost group reaches depth zero.
  TEST_NCCL(ncclGroupStart());
  TEST_NCCL(ncclGroupStart());
  TEST_NCCL(ncclAllGather(input, output, count, ncclFloat, comm, stream));
  TEST_NCCL(ncclGroupEnd());
  TEST_CUDA(cudaStreamSynchronize(stream));

  float first = -1.0f;
  TEST_CUDA(cudaMemcpy(&first, output, sizeof(first), cudaMemcpyDeviceToHost));
  if (first != 0.0f) {
    fprintf(stderr, "rank %d: COCCL primitive executed before outermost GroupEnd\n", rank);
    return 1;
  }

  pthread_barrier_wait(barrier);
  TEST_NCCL(ncclGroupEnd());
  TEST_CUDA(cudaStreamSynchronize(stream));
  TEST_CUDA(cudaMemcpy(&first, output, sizeof(first), cudaMemcpyDeviceToHost));
  if (first == 0.0f) {
    fprintf(stderr, "rank %d: COCCL primitive was not replayed at outermost GroupEnd\n", rank);
    return 1;
  }

  pthread_barrier_wait(barrier);
  // Mixing native work with a deferred COCCL primitive is deliberately
  // rejected. The failed group must also discard the deferred descriptor.
  TEST_NCCL(ncclGroupStart());
  TEST_NCCL(ncclAllGather(input, output, count, ncclFloat, comm, stream));
  TEST_NCCL(ncclBroadcast(input, output, count, ncclFloat, 0, comm, stream));
  ncclResult_t mixedResult = ncclGroupEnd();
  if (mixedResult != ncclInvalidUsage) {
    fprintf(stderr, "rank %d: mixed group returned %d instead of ncclInvalidUsage\n",
            rank, (int)mixedResult);
    return 1;
  }

  pthread_barrier_wait(barrier);
  TEST_NCCL(ncclAllGather(input, output, count, ncclFloat, comm, stream));
  TEST_CUDA(cudaStreamSynchronize(stream));

  TEST_CUDA(cudaFree(output));
  TEST_CUDA(cudaFree(input));
  TEST_CUDA(cudaStreamDestroy(stream));
  return 0;
}

int main() {
  constexpr int ranks = 2;
  int devices[ranks] = {0, 1};
  ncclComm_t comms[ranks] = {};
  pthread_barrier_t barrier;

  TEST_NCCL(ncclCommInitAll(comms, ranks, devices));
  if (pthread_barrier_init(&barrier, nullptr, ranks) != 0) {
    fprintf(stderr, "pthread_barrier_init failed\n");
    return 1;
  }

  int results[ranks] = {};
  std::thread workers[ranks];
  for (int rank = 0; rank < ranks; ++rank) {
    workers[rank] = std::thread([&, rank] {
      results[rank] = runRank(rank, comms[rank], &barrier);
    });
  }
  for (std::thread& worker : workers) worker.join();

  pthread_barrier_destroy(&barrier);
  for (int rank = 0; rank < ranks; ++rank) {
    TEST_NCCL(ncclCommDestroy(comms[rank]));
    if (results[rank] != 0) return results[rank];
  }
  printf("COCCL deferred group test passed\n");
  return 0;
}
