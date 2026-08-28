#include "core/runtime/coccl_primitive_dispatch.h"

#include "checks.h"
#include "core/memory/coccl_buffer_management.h"
#include "core/config/coccl_config.h"
#include "core/pipeline/coccl_frame_exchange.h"
#include "core/runtime/coccl_group_internal.h"
#include "core/runtime/coccl_prepared_call.h"
#include "runtime/coccl_runtime.h"
#include "collectives.h"
#include "comm.h"
#include "core/compression/compress.h"
#include "group.h"
#include "nccl.h"

#include <limits>
#include <vector>

namespace {

constexpr size_t kMetadataAlignment = 256;

struct cocclSendRecvCallState {
  cocclBufferHandle buffer;
  cocclCompressorView encoded = {};
  cocclCompressorFrameMetadata metadata = {};
  void* compressor = nullptr;
  size_t rawBytes = 0;
  size_t metadataOffset = 0;
  bool framed = false;
  bool readMetadata = false;
};

bool sendRecvLayout(size_t count, ncclDataType_t datatype,
                    size_t* rawBytes, size_t* metadataOffset,
                    size_t* allocationBytes) {
  const int typeBytes = ncclTypeSize(datatype);
  if (typeBytes <= 0 ||
      count > std::numeric_limits<size_t>::max() / (size_t)typeBytes) {
    return false;
  }
  *rawBytes = count * (size_t)typeBytes;
  if (*rawBytes > std::numeric_limits<size_t>::max() -
                      (kMetadataAlignment - 1)) {
    return false;
  }
  *metadataOffset =
      (*rawBytes + kMetadataAlignment - 1) & ~(kMetadataAlignment - 1);
  if (*metadataOffset > std::numeric_limits<size_t>::max() -
                            sizeof(cocclCompressorFrameMetadata)) {
    return false;
  }
  *allocationBytes =
      *metadataOffset + sizeof(cocclCompressorFrameMetadata);
  return true;
}

cocclCompressorFrameMetadata* deviceMetadata(
    const cocclSendRecvCallState& state) {
  return reinterpret_cast<cocclCompressorFrameMetadata*>(
      static_cast<char*>(state.buffer.ptr) + state.metadataOffset);
}

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
  ncclResult_t ret = ncclSuccess;
  std::vector<cocclSendRecvCallState> states(count);
  std::vector<cocclFrameExchange> metadataExchanges(count);
  std::vector<cocclFrameExchange> payloadExchanges(count);

  CUDACHECKGOTO(cudaSetDevice(calls[0].info.comm->cudaDev), ret, exit);
  for (size_t i = 0; i < count; ++i) {
    const cocclPreparedCall& prepared = calls[i];
    const cocclInfo& info = prepared.info;
    cocclSendRecvCallState& state = states[i];
    state.compressor = sendRecvCompressor(prepared);
    size_t allocationBytes = 0;
    if (!sendRecvLayout(info.count, info.datatype, &state.rawBytes,
                        &state.metadataOffset, &allocationBytes)) {
      ret = ncclInvalidArgument;
      goto exit;
    }
    NCCLCHECKGOTO(cocclGetBuffer(
                      info.comm, allocationBytes, info.stream, &state.buffer),
                  ret, exit);

    cocclCompressorFrameMetadata* metadata = deviceMetadata(state);
    state.framed = cocclCompressorSupports(
        state.compressor, cocclCompressorCapabilityFramed);
    if (info.func == ncclFuncSend) {
      const cocclCompressorView input = {
          const_cast<void*>(info.sendbuff), state.rawBytes, state.rawBytes,
          info.count, 1, info.datatype, nullptr, 0};
      state.encoded = {
          state.buffer.ptr, state.rawBytes, 0, 0, 1, ncclInt8,
          state.framed ? metadata : nullptr,
          state.framed ? state.rawBytes : 0};
      NCCLCHECKGOTO(ncclCompress(
                        state.compressor, input, &state.encoded,
                        info.comm->rank, info.stream),
                    ret, exit);

      if (state.framed) {
        state.readMetadata = true;
      } else {
        state.metadata.payloadBytes = state.encoded.bytes;
        state.metadata.encoding =
            state.encoded.datatype == COCCL_COMPRESSOR_RAW_PASSTHROUGH
            ? cocclCompressorFrameRaw
            : cocclCompressorFrameEncoded;
        state.metadata.reserved = 0;
        CUDACHECKGOTO(cudaMemcpyAsync(
                          metadata, &state.metadata, sizeof(state.metadata),
                          cudaMemcpyHostToDevice, info.stream),
                      ret, exit);
      }
    } else {
      state.readMetadata = true;
    }

    metadataExchanges[i] = {
        info.peer,
        info.func == ncclFuncSend ? metadata : nullptr,
        info.func == ncclFuncRecv ? metadata : nullptr,
        info.func == ncclFuncSend ? sizeof(*metadata) : 0,
        info.func == ncclFuncRecv ? sizeof(*metadata) : 0,
        sizeof(*metadata), info.comm, info.stream};
  }

  NCCLCHECKGOTO(cocclCommitFrameExchange(
                    metadataExchanges.data(), metadataExchanges.size(),
                    nullptr, nullptr),
                ret, exit);

  for (size_t i = 0; i < count; ++i) {
    if (!states[i].readMetadata) continue;
    CUDACHECKGOTO(cudaMemcpyAsync(
                      &states[i].metadata, deviceMetadata(states[i]),
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
    const cocclInfo& info = calls[i].info;
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
      input.frameMetadata = deviceMetadata(state);
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
                      state.compressor, input, &output, info.stream),
                  ret, exit);
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
