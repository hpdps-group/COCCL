#include "core/compression/compress.h"
#include "core/memory/coccl_buffer_management.h"
#include "core/pipeline/coccl_frame_exchange.h"
#include "core/runtime/coccl_comm.h"
#include "core/runtime/coccl_group_internal.h"
#include "core/runtime/coccl_prepared_call.h"
#include "core/training/coccl_training_assist.h"
#include "runtime/coccl_runtime.h"

#include "checks.h"
#include "collectives.h"
#include "comm.h"
#include "core.h"
#include "group.h"

#include <limits>
#include <vector>

namespace {

constexpr size_t kSendRecvMetadataAlignment = 256;

struct cocclSendRecvCallState {
  cocclBufferHandle buffer;
  cocclCompressorView encoded = {};
  cocclCompressorFrameMetadata metadata = {};
  size_t rawBytes = 0;
  size_t metadataOffset = 0;
  bool framed = false;
  bool readMetadata = false;
};

bool cocclSendRecvLayout(size_t count, ncclDataType_t datatype,
                         size_t* rawBytes, size_t* metadataOffset,
                         size_t* allocationBytes) {
  const int typeBytes = ncclTypeSize(datatype);
  if (typeBytes <= 0 ||
      count > std::numeric_limits<size_t>::max() / (size_t)typeBytes) {
    return false;
  }
  *rawBytes = count * (size_t)typeBytes;
  if (*rawBytes > std::numeric_limits<size_t>::max() -
                      (kSendRecvMetadataAlignment - 1)) {
    return false;
  }
  *metadataOffset =
      (*rawBytes + kSendRecvMetadataAlignment - 1) &
      ~(kSendRecvMetadataAlignment - 1);
  if (*metadataOffset > std::numeric_limits<size_t>::max() -
                            sizeof(cocclCompressorFrameMetadata)) {
    return false;
  }
  *allocationBytes =
      *metadataOffset + sizeof(cocclCompressorFrameMetadata);
  return true;
}

cocclCompressorFrameMetadata* cocclSendRecvDeviceMetadata(
    const cocclSendRecvCallState& state) {
  return reinterpret_cast<cocclCompressorFrameMetadata*>(
      static_cast<char*>(state.buffer.ptr) + state.metadataOffset);
}

bool cocclSendRecvForward(const cocclInfo& info) {
  return info.func == ncclFuncSend
      ? info.comm->rank < info.peer
      : info.comm->rank > info.peer;
}

cocclPolicyKey cocclDirectSendRecvPolicy(const cocclInfo& info) {
  return cocclTrainingAssistEnabled()
      ? cocclDirectionalPolicy(cocclOperation::SendRecv,
                               cocclSendRecvForward(info))
      : cocclDefaultPolicy(cocclOperation::SendRecv);
}

cocclPreparedCall cocclDirectSendRecvCall(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int peer, ncclFunc_t func, ncclComm_t comm,
    cudaStream_t stream) {
  cocclPreparedCall prepared;
  prepared.info.sendbuff = sendbuff;
  prepared.info.recvbuff = recvbuff;
  prepared.info.count = count;
  prepared.info.datatype = datatype;
  prepared.info.peer = peer;
  prepared.info.func = func;
  prepared.info.operation = cocclOperation::SendRecv;
  prepared.info.comm = comm;
  prepared.info.stream = stream;
  prepared.descriptor =
      cocclGetOperationDescriptor(cocclOperation::SendRecv);
  return prepared;
}

ncclResult_t cocclSubmitDirectSendRecv(cocclPreparedCall* prepared) {
  if (prepared->info.count == 0) {
    return ncclGroupDepth > 0
        ? cocclGroupEnqueueNative(&prepared->info)
        : cocclReplayNativeCall(prepared->info);
  }

  prepared->policy = cocclDirectSendRecvPolicy(prepared->info);
  ncclResult_t ret = cocclCommGetCompressor(
      prepared->info.comm, prepared->policy, &prepared->compressor);
  if (ret != ncclSuccess) {
    return ncclGroupDepth > 0
        ? cocclGroupEnqueueNative(&prepared->info)
        : ret;
  }
  return cocclEnqueuePreparedCall(prepared);
}

}  // namespace

ncclResult_t cocclExecuteSendRecvBatch(
    const cocclPreparedCall* calls, size_t count) {
  ncclResult_t ret = ncclSuccess;
  std::vector<cocclSendRecvCallState> states(count);
  std::vector<cocclFrameExchange> metadataExchanges(count);
  std::vector<cocclFrameExchange> payloadExchanges(count);

  {
    CUDACHECKGOTO(cudaSetDevice(calls[0].info.comm->cudaDev), ret, exit);

    for (size_t i = 0; i < count; ++i) {
      const cocclPreparedCall& prepared = calls[i];
      const cocclInfo& info = prepared.info;
      cocclSendRecvCallState& state = states[i];
      size_t allocationBytes = 0;
      if (!cocclSendRecvLayout(info.count, info.datatype, &state.rawBytes,
                               &state.metadataOffset, &allocationBytes)) {
        ret = ncclInvalidArgument;
        goto exit;
      }
      NCCLCHECKGOTO(cocclGetBufferForComm(
                        info.comm, info.comm, allocationBytes,
                        &state.buffer),
                    ret, exit);

      cocclCompressorFrameMetadata* deviceMetadata =
          cocclSendRecvDeviceMetadata(state);
      state.framed = cocclCompressorSupports(
          prepared.compressor, cocclCompressorCapabilityFramed);

      if (info.func == ncclFuncSend) {
        const cocclCompressorView input = {
            const_cast<void*>(info.sendbuff), state.rawBytes,
            state.rawBytes, info.count, 1, info.datatype, nullptr, 0};
        state.encoded = {
            state.buffer.ptr, state.rawBytes, 0, 0, 1, ncclInt8,
            state.framed ? deviceMetadata : nullptr,
            state.framed ? state.rawBytes : 0};
        NCCLCHECKGOTO(ncclCompress(
                          prepared.compressor, input, &state.encoded,
                          info.comm->rank, info.stream),
                      ret, exit);

        if (!state.framed) {
          state.metadata.payloadBytes = state.encoded.bytes;
          state.metadata.encoding =
              state.encoded.datatype == COCCL_COMPRESSOR_RAW_PASSTHROUGH
              ? cocclCompressorFrameRaw
              : cocclCompressorFrameEncoded;
          state.metadata.reserved = 0;
          CUDACHECKGOTO(cudaMemcpyAsync(
                            deviceMetadata, &state.metadata,
                            sizeof(state.metadata), cudaMemcpyHostToDevice,
                            info.stream),
                        ret, exit);
        } else {
          state.readMetadata = true;
        }
      } else {
        state.readMetadata = true;
      }

      metadataExchanges[i] = {
          info.peer,
          info.func == ncclFuncSend ? deviceMetadata : nullptr,
          info.func == ncclFuncRecv ? deviceMetadata : nullptr,
          info.func == ncclFuncSend
              ? sizeof(cocclCompressorFrameMetadata) : 0,
          info.func == ncclFuncRecv
              ? sizeof(cocclCompressorFrameMetadata) : 0,
          sizeof(cocclCompressorFrameMetadata), info.comm, info.stream};
    }

    NCCLCHECKGOTO(cocclCommitFrameExchange(
                      metadataExchanges.data(), metadataExchanges.size(),
                      nullptr, nullptr),
                  ret, exit);

    for (size_t i = 0; i < count; ++i) {
      if (!states[i].readMetadata) continue;
      CUDACHECKGOTO(cudaMemcpyAsync(
                        &states[i].metadata,
                        cocclSendRecvDeviceMetadata(states[i]),
                        sizeof(states[i].metadata), cudaMemcpyDeviceToHost,
                        calls[i].info.stream),
                    ret, exit);
    }
    for (size_t i = 0; i < count; ++i) {
      if (!states[i].readMetadata) continue;
      CUDACHECKGOTO(cudaStreamSynchronize(calls[i].info.stream), ret, exit);
    }

    for (size_t i = 0; i < count; ++i) {
      const cocclInfo& info = calls[i].info;
      cocclSendRecvCallState& state = states[i];
      if (!cocclFrameMetadataValid(state.metadata, state.rawBytes)) {
        ret = ncclInvalidUsage;
        goto exit;
      }
      payloadExchanges[i] = {
          info.peer,
          info.func == ncclFuncSend ? state.encoded.data : nullptr,
          info.func == ncclFuncRecv ? state.buffer.ptr : nullptr,
          info.func == ncclFuncSend
              ? (size_t)state.metadata.payloadBytes : 0,
          info.func == ncclFuncRecv
              ? (size_t)state.metadata.payloadBytes : 0,
          state.rawBytes, info.comm, info.stream};
    }

    NCCLCHECKGOTO(cocclCommitFrameExchange(
                      payloadExchanges.data(), payloadExchanges.size(),
                      nullptr, nullptr),
                  ret, exit);

    for (size_t i = 0; i < count; ++i) {
      const cocclPreparedCall& prepared = calls[i];
      const cocclInfo& info = prepared.info;
      cocclSendRecvCallState& state = states[i];
      if (info.func != ncclFuncRecv) continue;

      cocclCompressorView input = {};
      input.data = state.buffer.ptr;
      input.chunks = 1;
      if (state.framed) {
        input.capacityBytes = state.rawBytes;
        input.bytes = state.rawBytes;
        input.elements = state.rawBytes;
        input.datatype = ncclInt8;
        input.frameMetadata = cocclSendRecvDeviceMetadata(state);
        input.frameStrideBytes = state.rawBytes;
      } else {
        input.capacityBytes = (size_t)state.metadata.payloadBytes;
        input.bytes = (size_t)state.metadata.payloadBytes;
        input.elements = (size_t)state.metadata.payloadBytes;
        input.datatype =
            state.metadata.encoding == cocclCompressorFrameRaw
            ? COCCL_COMPRESSOR_RAW_PASSTHROUGH
            : ncclInt8;
      }
      cocclCompressorView output = {
          info.recvbuff, state.rawBytes, 0, info.count, 1,
          info.datatype, nullptr, 0};
      NCCLCHECKGOTO(ncclDecompress(
                        prepared.compressor, input, &output, info.stream),
                    ret, exit);
    }
  }

exit:
  for (size_t i = 0; i < count; ++i) {
    if (states[i].buffer.ptr == nullptr) continue;
    const ncclResult_t releaseResult =
        cocclReleaseBuffer(&states[i].buffer, calls[i].info.stream);
    if (ret == ncclSuccess) ret = releaseResult;
  }
  return ret;
}

NCCL_API(ncclResult_t, cocclSendComp, const void* sendbuff, size_t count,
  ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
ncclResult_t cocclSendComp(const void* sendbuff, size_t count,
  ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
  if (comm == nullptr) return ncclInvalidArgument;
  cocclPreparedCall prepared = cocclDirectSendRecvCall(
      sendbuff, nullptr, count, datatype, peer, ncclFuncSend, comm, stream);
  return cocclSubmitDirectSendRecv(&prepared);
}

NCCL_API(ncclResult_t, cocclRecvDecomp, void* recvbuff, size_t count,
  ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
ncclResult_t cocclRecvDecomp(void* recvbuff, size_t count,
  ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
  if (comm == nullptr) return ncclInvalidArgument;
  cocclPreparedCall prepared = cocclDirectSendRecvCall(
      nullptr, recvbuff, count, datatype, peer, ncclFuncRecv, comm, stream);
  return cocclSubmitDirectSendRecv(&prepared);
}

ncclResult_t cocclExecuteSendRecv(const cocclPreparedCall* prepared) {
  return cocclExecuteSendRecvBatch(prepared, 1);
}
