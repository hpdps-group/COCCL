/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"

#include <stdlib.h>

static int SendRecvPeer(int rank, int nranks, int direction) {
  const char* crossNode = getenv("COCCL_SENDRECV_CROSS_NODE");
  if (crossNode != nullptr && crossNode[0] == '1') {
    return (rank + nranks / 2) % nranks;
  }
  return (rank + direction + nranks) % nranks;
}

void SendRecvGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, int nranks) {
  *sendcount = count;
  *recvcount = count;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = *sendcount;
}

testResult_t SendRecvInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    __nv_bfloat16* hostdata = (__nv_bfloat16*)malloc(sendcount*wordSize(type));
    const char* randomInput = getenv("COCCL_SENDRECV_RANDOM_INPUT");
    if (randomInput != nullptr && randomInput[0] == '1') {
      uint32_t state = 0x9e3779b9u ^ static_cast<uint32_t>(rank + 1);
      unsigned char* bytes = reinterpret_cast<unsigned char*>(hostdata);
      for (size_t byte = 0; byte < sendcount * wordSize(type); ++byte) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        bytes[byte] = static_cast<unsigned char>(state);
      }
    } else {
      for (size_t ii = 0; ii < sendcount; ++ii) {
        hostdata[ii] = (__nv_bfloat16)ii;
        if (ii % 128 == 64) {
          hostdata[ii] = static_cast<float>((ii / 128 + 1) * 256);
        }
      }
    }
    CUDACHECK(cudaMemcpy(data, hostdata, sendcount*wordSize(type), cudaMemcpyHostToDevice));
    free(hostdata);
    // TESTCHECK(InitData(data, sendcount, rank*sendcount, type, ncclSum, rep, 1, 0));
    int peer = SendRecvPeer(rank, nranks, -1);
    TESTCHECK(InitData(args->expected[i], recvcount, peer*recvcount, type, ncclSum, rep, 1, 0));
    CUDACHECK(cudaDeviceSynchronize());
  }
  // We don't support in-place sendrecv
  args->reportErrors = in_place ? 0 : 1;
  return testSuccess;
}

void SendRecvGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = 1;
  *busBw = baseBw * factor;
}

testResult_t SendRecvRunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  int rank;
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  int previous = SendRecvPeer(rank, nRanks, -1);
  int next = SendRecvPeer(rank, nRanks, 1);

  const char* sameStreamBidirectional =
      getenv("COCCL_SENDRECV_SAME_STREAM_BIDIRECTIONAL");
  const bool runSameStreamBidirectional =
      sameStreamBidirectional != nullptr && sameStreamBidirectional[0] == '1';
  const char* batchStress = getenv("COCCL_SENDRECV_BATCH_STRESS");
  const bool runBatchStress =
      batchStress != nullptr && batchStress[0] == '1';
  if (!runSameStreamBidirectional && !runBatchStress) {
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(cocclRecvDecomp(
        recvbuff, count, type, previous, comm, stream));
    NCCLCHECK(cocclSendComp(
        sendbuff, count, type, next, comm, stream));
    NCCLCHECK(ncclGroupEnd());
    return testSuccess;
  }

  const size_t typeBytes = wordSize(type);
  if (runSameStreamBidirectional) {
    const size_t firstCount = count / 2;
    const size_t secondCount = count - firstCount;
    const size_t secondOffset = firstCount * typeBytes;

    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(cocclRecvDecomp(
        recvbuff, firstCount, type, previous, comm, stream));
    NCCLCHECK(cocclSendComp(
        sendbuff, firstCount, type, next, comm, stream));
    NCCLCHECK(cocclRecvDecomp(
        static_cast<char*>(recvbuff) + secondOffset, secondCount, type,
        next, comm, stream));
    NCCLCHECK(cocclSendComp(
        static_cast<char*>(sendbuff) + secondOffset, secondCount, type,
        previous, comm, stream));
    NCCLCHECK(ncclGroupEnd());
    return testSuccess;
  }

  static thread_local cudaStream_t secondaryStream = nullptr;
  static thread_local cudaEvent_t secondaryDone = nullptr;
  if (secondaryStream == nullptr) {
    CUDACHECK(cudaStreamCreateWithFlags(&secondaryStream,
                                         cudaStreamNonBlocking));
    CUDACHECK(cudaEventCreateWithFlags(&secondaryDone,
                                        cudaEventDisableTiming));
  }

  const size_t segment = count / 4;
  const size_t segmentCounts[4] = {
      segment, segment, segment, count - 3 * segment};
  const size_t segmentOffsets[4] = {
      0, segment, 2 * segment, 3 * segment};

  NCCLCHECK(ncclGroupStart());
  for (int message = 0; message < 4; ++message) {
    const bool forward = (message & 1) == 0;
    const int recvPeer = forward ? previous : next;
    const int sendPeer = forward ? next : previous;
    cudaStream_t messageStream = forward ? stream : secondaryStream;
    const size_t offset = segmentOffsets[message] * typeBytes;
    NCCLCHECK(cocclRecvDecomp(
        static_cast<char*>(recvbuff) + offset, segmentCounts[message], type,
        recvPeer, comm, messageStream));
    NCCLCHECK(cocclSendComp(
        static_cast<char*>(sendbuff) + offset, segmentCounts[message], type,
        sendPeer, comm, messageStream));
  }
  NCCLCHECK(ncclGroupEnd());
  CUDACHECK(cudaEventRecord(secondaryDone, secondaryStream));
  CUDACHECK(cudaStreamWaitEvent(stream, secondaryDone, 0));
  return testSuccess;
}

struct testColl sendRecvTest = {
  "SendRecv",
  SendRecvGetCollByteCount,
  SendRecvInitData,
  SendRecvGetBw,
  SendRecvRunColl
};

void SendRecvGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  SendRecvGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t SendRecvRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &sendRecvTest;
  ncclDataType_t *run_types;
  ncclRedOp_t *run_ops;
  const char **run_typenames, **run_opnames;
  int type_count, op_count;

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = test_typenum;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  if ((int)op != -1) {
    op_count = 1;
    run_ops = &op;
    run_opnames = &opName;
  } else {
    op_count = test_opnum;
    run_ops = test_ops;
    run_opnames = test_opnames;
  }

  for (int i=0; i<type_count; i++) {
    for (int j=0; j<op_count; j++) {
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], run_ops[j], run_opnames[j], -1));
    }
  }
  return testSuccess;
}

struct testEngine sendRecvEngine = {
  SendRecvGetBuffSize,
  SendRecvRunTest
};

#pragma weak ncclTestEngine=sendRecvEngine
