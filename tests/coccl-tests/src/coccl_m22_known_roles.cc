#include "nccl.h"

#include <cuda_runtime.h>
#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <unistd.h>

namespace {

int worldRank = 0;
constexpr size_t kMessageElements = 1ULL << 18;
constexpr size_t kDpAllGatherChunkElements = kMessageElements / 2;
constexpr size_t kDpReduceScatterChunkElements = kMessageElements;
constexpr size_t kBufferElements = 2 * kMessageElements;

void fail(const char* api, int code, int line) {
  std::fprintf(stderr, "rank %d line %d: %s failed with %d\n",
               worldRank, line, api, code);
  MPI_Abort(MPI_COMM_WORLD, code == 0 ? 1 : code);
}

#define TEST_MPICHECK(call) \
  do { int result = (call); if (result != MPI_SUCCESS) \
    fail(#call, result, __LINE__); } while (0)
#define TEST_NCCLCHECK(call) \
  do { ncclResult_t result = (call); if (result != ncclSuccess) \
    fail(#call, (int)result, __LINE__); } while (0)
#define TEST_CUDACHECK(call) \
  do { cudaError_t result = (call); if (result != cudaSuccess) \
    fail(#call, (int)result, __LINE__); } while (0)

ncclComm_t split(ncclComm_t world, int color, int key) {
  ncclComm_t child = nullptr;
  TEST_NCCLCHECK(ncclCommSplit(world, color, key, &child, nullptr));
  return child;
}

void runCycle(ncclComm_t dp, ncclComm_t tp, ncclComm_t pp,
              float* send, float* recv) {
  TEST_NCCLCHECK(ncclAllGather(
      send, recv, kDpAllGatherChunkElements, ncclFloat32, dp, nullptr));
  for (int call = 0; call < 6; ++call) {
    TEST_NCCLCHECK(ncclAllReduce(
        send, recv, kMessageElements, ncclFloat32, ncclSum, tp, nullptr));
  }

  int ppRank = 0;
  TEST_NCCLCHECK(ncclCommUserRank(pp, &ppRank));
  TEST_NCCLCHECK(ncclGroupStart());
  for (int microbatch = 0; microbatch < 4; ++microbatch) {
    if (ppRank == 0) {
      TEST_NCCLCHECK(ncclSend(
          send, kMessageElements, ncclFloat32, 1, pp, nullptr));
      TEST_NCCLCHECK(ncclRecv(
          recv, kMessageElements, ncclFloat32, 1, pp, nullptr));
    } else {
      TEST_NCCLCHECK(ncclRecv(
          recv, kMessageElements, ncclFloat32, 0, pp, nullptr));
      TEST_NCCLCHECK(ncclSend(
          send, kMessageElements, ncclFloat32, 0, pp, nullptr));
    }
  }
  TEST_NCCLCHECK(ncclGroupEnd());

  TEST_NCCLCHECK(ncclReduceScatter(
      send, recv, kDpReduceScatterChunkElements, ncclFloat32, ncclSum,
      dp, nullptr));
  usleep(1000);
}

}  // namespace

int main(int argc, char** argv) {
  TEST_MPICHECK(MPI_Init(&argc, &argv));
  int worldSize = 0;
  TEST_MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
  TEST_MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
  if (worldSize != 8) fail("world size must be 8", worldSize, __LINE__);

  MPI_Comm localComm;
  int localRank = 0;
  TEST_MPICHECK(MPI_Comm_split_type(
      MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank,
      MPI_INFO_NULL, &localComm));
  TEST_MPICHECK(MPI_Comm_rank(localComm, &localRank));
  TEST_CUDACHECK(cudaSetDevice(localRank));

  ncclUniqueId uniqueId;
  if (worldRank == 0) TEST_NCCLCHECK(ncclGetUniqueId(&uniqueId));
  TEST_MPICHECK(MPI_Bcast(
      &uniqueId, sizeof(uniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));

  ncclComm_t world = nullptr;
  TEST_NCCLCHECK(ncclCommInitRank(
      &world, worldSize, uniqueId, worldRank));
  ncclComm_t dp = split(world, worldRank % 4, worldRank / 4);
  ncclComm_t tp = split(world, worldRank / 2, worldRank % 2);
  ncclComm_t pp = split(
      world, (worldRank / 4) * 2 + worldRank % 2,
      (worldRank % 4) / 2);

  std::printf("COCCL_M22_COMM world_rank=%d kind=dp comm=%p expected=DP\n",
              worldRank, dp);
  std::printf("COCCL_M22_COMM world_rank=%d kind=tp comm=%p expected=TP\n",
              worldRank, tp);
  std::printf("COCCL_M22_COMM world_rank=%d kind=pp comm=%p expected=PP\n",
              worldRank, pp);
  std::fflush(stdout);

  float* send = nullptr;
  float* recv = nullptr;
  TEST_CUDACHECK(cudaMalloc(&send, kBufferElements * sizeof(float)));
  TEST_CUDACHECK(cudaMalloc(&recv, kBufferElements * sizeof(float)));
  TEST_CUDACHECK(cudaMemset(send, 0, kBufferElements * sizeof(float)));
  TEST_CUDACHECK(cudaMemset(recv, 0, kBufferElements * sizeof(float)));

  for (int iteration = 0; iteration < 10; ++iteration) {
    runCycle(dp, tp, pp, send, recv);
  }
  TEST_CUDACHECK(cudaDeviceSynchronize());
  std::printf("COCCL_M22_PHASE world_rank=%d phase=observed_10\n", worldRank);
  std::fflush(stdout);

  for (int iteration = 10; iteration < 16; ++iteration) {
    runCycle(dp, tp, pp, send, recv);
  }
  TEST_CUDACHECK(cudaDeviceSynchronize());
  std::printf("COCCL_M22_PHASE world_rank=%d phase=classification_expected\n",
              worldRank);
  std::fflush(stdout);

  // Register the constant-size auxiliary communicator only after the known
  // roles commit. Its isolated trace has no DP/TP ordering evidence and must
  // therefore remain Unknown.
  ncclComm_t ambiguous = split(world, worldRank / 2, worldRank % 2);
  std::printf(
      "COCCL_M22_COMM world_rank=%d kind=ambiguous comm=%p expected=Unknown\n",
      worldRank, ambiguous);
  std::fflush(stdout);
  for (int iteration = 0; iteration < 10; ++iteration) {
    for (int call = 0; call < 8; ++call) {
      TEST_NCCLCHECK(ncclAllReduce(
          send, recv, kMessageElements, ncclFloat32, ncclSum,
          ambiguous, nullptr));
    }
  }
  TEST_CUDACHECK(cudaDeviceSynchronize());
  std::printf("COCCL_M22_PHASE world_rank=%d phase=unknown_expected\n",
              worldRank);
  std::fflush(stdout);

  for (int iteration = 0; iteration < 2; ++iteration) {
    runCycle(dp, tp, pp, send, recv);
    TEST_NCCLCHECK(ncclAllReduce(
        send, recv, kMessageElements, ncclFloat32, ncclSum,
        ambiguous, nullptr));
  }
  TEST_CUDACHECK(cudaDeviceSynchronize());
  std::printf("COCCL_M22_PHASE world_rank=%d phase=stable_complete\n",
              worldRank);
  std::fflush(stdout);

  TEST_CUDACHECK(cudaFree(recv));
  TEST_CUDACHECK(cudaFree(send));
  TEST_NCCLCHECK(ncclCommDestroy(ambiguous));
  TEST_NCCLCHECK(ncclCommDestroy(pp));
  TEST_NCCLCHECK(ncclCommDestroy(tp));
  TEST_NCCLCHECK(ncclCommDestroy(dp));
  TEST_NCCLCHECK(ncclCommDestroy(world));
  TEST_MPICHECK(MPI_Comm_free(&localComm));
  TEST_MPICHECK(MPI_Finalize());
  return 0;
}
