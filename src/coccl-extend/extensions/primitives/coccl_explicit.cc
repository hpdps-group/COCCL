#include "core/runtime/coccl_prepared_call.h"

#include "core.h"
#include "nccl.h"

namespace {

cocclInfo collectiveInfo(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclFunc_t func,
    cocclOperation operation, ncclComm_t comm, cudaStream_t stream) {
  cocclInfo info;
  info.sendbuff = sendbuff;
  info.recvbuff = recvbuff;
  info.count = count;
  info.datatype = datatype;
  info.op = op;
  info.func = func;
  info.operation = operation;
  info.comm = comm;
  info.stream = stream;
  return info;
}

}  // namespace

NCCL_API(ncclResult_t, cocclAllGatherComp, const void*, void*, size_t,
         ncclDataType_t, ncclComm_t, cudaStream_t);
ncclResult_t cocclAllGatherComp(
    const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  const cocclInfo info = collectiveInfo(
      sendbuff, recvbuff, sendcount, datatype, ncclSum, ncclFuncAllGather,
      cocclOperation::AllGather, comm, stream);
  return cocclEnqueueExplicitCall(&info, cocclAlgorithmNone);
}

NCCL_API(ncclResult_t, cocclAllToAllComp, const void*, void*, size_t,
         ncclDataType_t, ncclComm_t, cudaStream_t);
ncclResult_t cocclAllToAllComp(
    const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  const cocclInfo info = collectiveInfo(
      sendbuff, recvbuff, sendcount, datatype, ncclSum, ncclNumFuncs,
      cocclOperation::AllToAll, comm, stream);
  return cocclEnqueueExplicitCall(&info, cocclAlgorithmNone);
}

NCCL_API(ncclResult_t, cocclReduceScatterCompOneShot, const void*, void*,
         size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
ncclResult_t cocclReduceScatterCompOneShot(
    const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    cudaStream_t stream) {
  const cocclInfo info = collectiveInfo(
      sendbuff, recvbuff, recvcount, datatype, op, ncclFuncReduceScatter,
      cocclOperation::ReduceScatter, comm, stream);
  return cocclEnqueueExplicitCall(
      &info, cocclAlgorithmReduceScatterOneShot);
}

NCCL_API(ncclResult_t, cocclReduceScatterCompTwoShot, const void*, void*,
         size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
ncclResult_t cocclReduceScatterCompTwoShot(
    const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    cudaStream_t stream) {
  const cocclInfo info = collectiveInfo(
      sendbuff, recvbuff, recvcount, datatype, op, ncclFuncReduceScatter,
      cocclOperation::ReduceScatter, comm, stream);
  return cocclEnqueueExplicitCall(
      &info, cocclAlgorithmReduceScatterTwoShot);
}

NCCL_API(ncclResult_t, cocclAllReduceCompOneShot, const void*, void*, size_t,
         ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
ncclResult_t cocclAllReduceCompOneShot(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    cudaStream_t stream) {
  const cocclInfo info = collectiveInfo(
      sendbuff, recvbuff, count, datatype, op, ncclFuncAllReduce,
      cocclOperation::AllReduce, comm, stream);
  return cocclEnqueueExplicitCall(&info, cocclAlgorithmAllReduceOneShot);
}

NCCL_API(ncclResult_t, cocclAllReduceCompTwoShot, const void*, void*, size_t,
         ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
ncclResult_t cocclAllReduceCompTwoShot(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    cudaStream_t stream) {
  const cocclInfo info = collectiveInfo(
      sendbuff, recvbuff, count, datatype, op, ncclFuncAllReduce,
      cocclOperation::AllReduce, comm, stream);
  return cocclEnqueueExplicitCall(&info, cocclAlgorithmAllReduceTwoShot);
}

NCCL_API(ncclResult_t, cocclAllReduceCompTripleShot, const void*, void*,
         size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
ncclResult_t cocclAllReduceCompTripleShot(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    cudaStream_t stream) {
  const cocclInfo info = collectiveInfo(
      sendbuff, recvbuff, count, datatype, op, ncclFuncAllReduce,
      cocclOperation::AllReduce, comm, stream);
  return cocclEnqueueExplicitCall(&info, cocclAlgorithmAllReduceTripleShot);
}
