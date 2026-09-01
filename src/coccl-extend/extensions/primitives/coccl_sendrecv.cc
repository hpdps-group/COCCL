#include "core/runtime/coccl_primitive_dispatch.h"

#include "core/pipeline/coccl_pipeline.h"
#include "core/runtime/coccl_group_internal.h"
#include "core/runtime/coccl_prepared_call.h"
#include "runtime/coccl_runtime.h"
#include "comm.h"
#include "group.h"
#include "nccl.h"

#include <array>
#include <vector>

namespace {

cocclInfo directCall(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int peer, ncclFunc_t func, ncclComm_t comm,
    cudaStream_t stream) {
  cocclInfo info;
  info.sendbuff = sendbuff;
  info.recvbuff = recvbuff;
  info.count = count;
  info.datatype = datatype;
  info.peer = peer;
  info.func = func;
  info.operation = cocclOperation::SendRecv;
  info.comm = comm;
  info.stream = stream;
  return info;
}

ncclResult_t submitDirect(const cocclInfo& info) {
  if (info.count == 0) {
    return ncclGroupDepth > 0
        ? cocclGroupEnqueueNative(&info)
        : cocclReplayNativeCall(info);
  }
  return cocclEnqueueExplicitCall(&info, cocclAlgorithmNone);
}

void* sendRecvCompressor(const cocclPreparedCall& prepared) {
  const cocclInfo& info = prepared.info;
  const cocclCompressionScope scope =
      info.comm->rankToNode[info.peer] == info.comm->node
      ? cocclCompressionScope::Intra
      : cocclCompressionScope::Inter;
  return prepared.compressors.get(scope);
}

}  // namespace

ncclResult_t cocclExecuteSendRecvBatch(
    const cocclPreparedCall* calls, size_t count) {
  std::vector<std::array<cocclPipelineStage, 2>> stages(count);
  std::vector<cocclPipelineSpec> specs(count);
  for (size_t i = 0; i < count; ++i) {
    const cocclInfo& info = calls[i].info;
    void* compressor = sendRecvCompressor(calls[i]);
    const cocclPipelineStage exchange = cocclPipelineSendRecv(
        info.comm, info.peer,
        info.func == ncclFuncSend ? cocclPipelineSend : cocclPipelineRecv,
        compressor);
    if (info.func == ncclFuncSend) {
      stages[i] = {cocclPipelineCompress(compressor), exchange};
    } else {
      stages[i] = {exchange, cocclPipelineDecompress()};
    }
    const void* buffer = info.func == ncclFuncSend
        ? info.sendbuff : info.recvbuff;
    specs[i] = {
        info.func == ncclFuncSend ? "send" : "recv",
        buffer, const_cast<void*>(buffer), info.count, 1, info.datatype,
        info.comm, info.stream, stages[i].data(), 2,
        cocclPipelineInPlaceSameBuffer,
        cocclPipelineInputContiguous, 0};
  }
  return cocclRunPipelineBatch(specs.data(), specs.size());
}

NCCL_API(ncclResult_t, cocclSendComp, const void* sendbuff, size_t count,
  ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
ncclResult_t cocclSendComp(const void* sendbuff, size_t count,
  ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
  if (comm == nullptr) return ncclInvalidArgument;
  const cocclInfo info = directCall(
      sendbuff, nullptr, count, datatype, peer, ncclFuncSend, comm, stream);
  return submitDirect(info);
}

NCCL_API(ncclResult_t, cocclRecvDecomp, void* recvbuff, size_t count,
  ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
ncclResult_t cocclRecvDecomp(void* recvbuff, size_t count,
  ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
  if (comm == nullptr) return ncclInvalidArgument;
  const cocclInfo info = directCall(
      nullptr, recvbuff, count, datatype, peer, ncclFuncRecv, comm, stream);
  return submitDirect(info);
}

ncclResult_t cocclExecuteSendRecv(const cocclPreparedCall* prepared) {
  return cocclExecuteSendRecvBatch(prepared, 1);
}
