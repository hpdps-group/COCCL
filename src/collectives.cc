/*************************************************************************
 * Copyright (c) 2015-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "argcheck.h" // Need some checks here since we access comm
#include "collectives.h"
#include "runtime/coccl_group.h"
#include "runtime/coccl_runtime.h"
#include "enqueue.h"
#include "group.h"
#include "nccl.h"

 NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
     ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
 ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
     ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
   size_t msgsize = sendcount * ncclTypeSize(datatype);
   cocclInfo cocclCall;
   cocclCall.sendbuff = sendbuff;
   cocclCall.recvbuff = recvbuff;
   cocclCall.count = sendcount;
   cocclCall.datatype = datatype;
   cocclCall.func = ncclFuncAllGather;
   cocclCall.operation = cocclOperation::AllGather;
   cocclCall.comm = comm;
   cocclCall.stream = stream;
   bool isEnqueued = false;
   NCCLCHECK(cocclEnqueueCheck(&cocclCall, &isEnqueued));
   if (isEnqueued) return ncclSuccess;

   // Just pass the size of one message and not the total bytes sent/received.
   constexpr nvtxPayloadSchemaEntry_t AllGatherSchema[] = {
     {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"}
   };
  
   NVTX3_FUNC_WITH_PARAMS(AllGather, AllGatherSchema, msgsize)
 
   struct ncclInfo info = { ncclFuncAllGather, "AllGather",
     sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
     ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
   return ncclEnqueueCheck(&info);
 }
 
 NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
     ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
 ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
     ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
    cocclInfo cocclCall;
    cocclCall.sendbuff = sendbuff;
    cocclCall.recvbuff = recvbuff;
    cocclCall.count = count;
    cocclCall.datatype = datatype;
    cocclCall.op = op;
    cocclCall.func = ncclFuncAllReduce;
    cocclCall.operation = cocclOperation::AllReduce;
    cocclCall.comm = comm;
    cocclCall.stream = stream;
    bool isEnqueued = false;
    NCCLCHECK(cocclEnqueueCheck(&cocclCall, &isEnqueued));
    if (isEnqueued) return ncclSuccess;

    struct NvtxParamsAllReduce {
      size_t bytes;
      ncclRedOp_t op;
     };
     // Just pass the size of one message and not the total bytes sent/received.
     static constexpr nvtxPayloadSchemaEntry_t AllReduceSchema[] = {
       {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"},
       {0, NVTX_PAYLOAD_ENTRY_NCCL_REDOP, "Reduction operation", nullptr, 0,
         offsetof(NvtxParamsAllReduce, op)}
     };
     NvtxParamsAllReduce payload{count * ncclTypeSize(datatype), op};
     NVTX3_FUNC_WITH_PARAMS(AllReduce, AllReduceSchema, payload)
    struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
      sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
      ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
    return ncclEnqueueCheck(&info);
}
 
 NCCL_API(ncclResult_t, ncclBroadcast, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
     ncclComm_t comm, cudaStream_t stream);
 ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
     ncclComm_t comm, cudaStream_t stream) {
   struct NvtxParamsBroadcast {
     size_t bytes;
     int root;
   };
   constexpr nvtxPayloadSchemaEntry_t BroadcastSchema[] = {
     {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Bytes"},
     {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Root", nullptr, 0, offsetof(NvtxParamsBroadcast, root)}
   };
   NvtxParamsBroadcast payload{count * ncclTypeSize(datatype), root};
   NVTX3_FUNC_WITH_PARAMS(Broadcast, BroadcastSchema, payload)
 
   struct ncclInfo info = { ncclFuncBroadcast, "Broadcast",
     sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
     BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS };
   NCCLCHECK(ncclEnqueueCheck(&info));
   return ncclSuccess;
 }
 /* Deprecated original "in place" function, similar to MPI */
 NCCL_API(ncclResult_t, ncclBcast, void* buff, size_t count, ncclDataType_t datatype, int root,
     ncclComm_t comm, cudaStream_t stream);
 ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
     ncclComm_t comm, cudaStream_t stream) {
   NCCLCHECK(ncclBroadcast(buff, buff, count, datatype, root, comm, stream));
   return ncclSuccess;
 }
 
 NCCL_API(ncclResult_t, ncclReduce, const void* sendbuff, void* recvbuff, size_t count,
     ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
 ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
     ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
   struct NvtxParamsReduce {
     size_t bytes;
     int root;
     ncclRedOp_t op;
   };
   constexpr nvtxPayloadSchemaEntry_t ReduceSchema[] = {
     {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"},
     {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Root", nullptr, 0, offsetof(NvtxParamsReduce, root)},
     {0, NVTX_PAYLOAD_ENTRY_NCCL_REDOP, "Reduction operation", nullptr, 0,
       offsetof(NvtxParamsReduce, op)}
   };
   NvtxParamsReduce payload{count * ncclTypeSize(datatype), root, op};
   NVTX3_FUNC_WITH_PARAMS(Reduce, ReduceSchema, payload)
 
   struct ncclInfo info = { ncclFuncReduce, "Reduce",
     sendbuff, recvbuff, count, datatype, op, root, comm, stream, /* Args */
     REDUCE_CHUNKSTEPS, REDUCE_SLICESTEPS };
   NCCLCHECK(ncclEnqueueCheck(&info));
   return ncclSuccess;
 }
 
 NCCL_API(ncclResult_t, ncclReduceScatter, const void* sendbuff, void* recvbuff, size_t recvcount,
     ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
 ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
     ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
   size_t msgsize = recvcount * ncclTypeSize(datatype);
   cocclInfo cocclCall;
   cocclCall.sendbuff = sendbuff;
   cocclCall.recvbuff = recvbuff;
   cocclCall.count = recvcount;
   cocclCall.datatype = datatype;
   cocclCall.op = op;
   cocclCall.func = ncclFuncReduceScatter;
   cocclCall.operation = cocclOperation::ReduceScatter;
   cocclCall.comm = comm;
   cocclCall.stream = stream;
   bool isEnqueued = false;
   NCCLCHECK(cocclEnqueueCheck(&cocclCall, &isEnqueued));
   if (isEnqueued) return ncclSuccess;

   struct NvtxParamsReduceScatter {
     size_t bytes;
     ncclRedOp_t op;
   };
   constexpr nvtxPayloadSchemaEntry_t ReduceScatterSchema[] = {
     {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"},
     {0, NVTX_PAYLOAD_ENTRY_NCCL_REDOP, "Reduction operation", nullptr, 0,
       offsetof(NvtxParamsReduceScatter, op)}
   };
   NvtxParamsReduceScatter payload{msgsize, op};
   NVTX3_FUNC_WITH_PARAMS(ReduceScatter, ReduceScatterSchema, payload)
   struct ncclInfo info = { ncclFuncReduceScatter, "ReduceScatter",
     sendbuff, recvbuff, recvcount, datatype, op, 0, comm, stream, /* Args */
     REDUCESCATTER_CHUNKSTEPS, REDUCESCATTER_SLICESTEPS };
   return ncclEnqueueCheck(&info);
 }
 
 struct NvtxParamsSendRecv {
     size_t bytes;
     int peer;
 };
 constexpr const nvtxPayloadSchemaEntry_t SendRecvSchema[] = {
     {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Bytes"},
     {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Peer rank", nullptr, 0, offsetof(NvtxParamsSendRecv, peer)}
 };
 
 NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
     ncclComm_t comm, cudaStream_t stream);
 ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
     ncclComm_t comm, cudaStream_t stream) {
   size_t msgsize = count * ncclTypeSize(datatype);
   NvtxParamsSendRecv payload{msgsize, peer};
   NVTX3_FUNC_WITH_PARAMS(Send, SendRecvSchema, payload)
   cocclInfo cocclCall;
   cocclCall.sendbuff = sendbuff;
   cocclCall.count = count;
   cocclCall.datatype = datatype;
   cocclCall.peer = peer;
   cocclCall.func = ncclFuncSend;
   cocclCall.operation = cocclOperation::SendRecv;
   cocclCall.comm = comm;
   cocclCall.stream = stream;
   bool isEnqueued = false;
   NCCLCHECK(cocclEnqueueCheck(&cocclCall, &isEnqueued));
   if (isEnqueued) return ncclSuccess;
   struct ncclInfo info = { ncclFuncSend, "Send",
     NULL, (void*)sendbuff, count, datatype, ncclSum, peer, comm, stream,
     1, 1 };
   return ncclEnqueueCheck(&info);
 }
 
 NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
     ncclComm_t comm, cudaStream_t stream);
 ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
     ncclComm_t comm, cudaStream_t stream) {
   size_t msgsize = count * ncclTypeSize(datatype);
   NvtxParamsSendRecv payload{msgsize, peer};
   NVTX3_FUNC_WITH_PARAMS(Recv, SendRecvSchema, payload)

   cocclInfo cocclCall;
   cocclCall.recvbuff = recvbuff;
   cocclCall.count = count;
   cocclCall.datatype = datatype;
   cocclCall.peer = peer;
   cocclCall.func = ncclFuncRecv;
   cocclCall.operation = cocclOperation::SendRecv;
   cocclCall.comm = comm;
   cocclCall.stream = stream;
   bool isEnqueued = false;
   NCCLCHECK(cocclEnqueueCheck(&cocclCall, &isEnqueued));
   if (isEnqueued) return ncclSuccess;
   struct ncclInfo info = { ncclFuncRecv, "Recv",
     NULL, recvbuff, count, datatype, ncclSum, peer, comm, stream,
     1, 1 };
   return ncclEnqueueCheck(&info);
 }
 
 NCCL_API(ncclResult_t, ncclAllToAll, const void* sendbuff, void* recvbuff, size_t sendcount,
     ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
 ncclResult_t ncclAllToAll(const void* sendbuff, void* recvbuff, size_t sendcount,
     ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
   cocclInfo cocclCall;
   cocclCall.sendbuff = sendbuff;
   cocclCall.recvbuff = recvbuff;
   cocclCall.count = sendcount;
   cocclCall.datatype = datatype;
   cocclCall.operation = cocclOperation::AllToAll;
   cocclCall.comm = comm;
   cocclCall.stream = stream;
   bool isEnqueued = false;
   NCCLCHECK(cocclEnqueueCheck(&cocclCall, &isEnqueued));
   if (isEnqueued) return ncclSuccess;

   NCCLCHECK(ncclGroupStart());
   for (size_t r = 0; r < comm->nRanks; r++) {
     char* r_sendbuf = (char*)sendbuff +
         r * sendcount * ncclTypeSize(datatype);
     char* r_recvbuf = (char*)recvbuff +
         r * sendcount * ncclTypeSize(datatype);

     cocclInfo recvCall;
     recvCall.recvbuff = r_recvbuf;
     recvCall.count = sendcount;
     recvCall.datatype = datatype;
     recvCall.peer = (int)r;
     recvCall.func = ncclFuncRecv;
     recvCall.operation = cocclOperation::SendRecv;
     recvCall.comm = comm;
     recvCall.stream = stream;
     NCCLCHECK(cocclReplayNativeCall(recvCall));

     cocclInfo sendCall;
     sendCall.sendbuff = r_sendbuf;
     sendCall.count = sendcount;
     sendCall.datatype = datatype;
     sendCall.peer = (int)r;
     sendCall.func = ncclFuncSend;
     sendCall.operation = cocclOperation::SendRecv;
     sendCall.comm = comm;
     sendCall.stream = stream;
     NCCLCHECK(cocclReplayNativeCall(sendCall));
   }
   NCCLCHECK(ncclGroupEnd());
   return ncclSuccess;
 }
 
 
