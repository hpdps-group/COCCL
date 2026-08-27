#include "runtime/coccl_runtime.h"

#include "core/tuning/coccl_autotune.h"
#include "core/config/coccl_config.h"
#include "core/runtime/coccl_group_internal.h"
#include "core/runtime/coccl_prepared_call.h"
#include "core/runtime/coccl_primitive_dispatch.h"
#include "core/training/coccl_training_assist.h"
#include "collectives.h"
#include "comm.h"
#include "core/compression/compress.h"
#include "debug.h"
#include "group.h"

#include <limits>

namespace {

__thread bool callerGuardActive = false;

bool floatingDatatypeSupported(ncclDataType_t datatype) {
  if (datatype == ncclFloat16 || datatype == ncclFloat32) return true;
#if defined(__CUDA_BF16_TYPES_EXIST__)
  return datatype == ncclBfloat16;
#else
  return false;
#endif
}

bool compressorDatatypeSupported(ncclDataType_t datatype,
                                 void* compressor) {
  if (floatingDatatypeSupported(datatype)) return true;
  if (datatype != ncclInt8 && datatype != ncclInt32 &&
      datatype != ncclInt64) {
    return false;
  }
  return cocclCompressorSupports(
      compressor, cocclCompressorCapabilityBytewiseLossless);
}

bool shapeSupported(const cocclInfo& info,
                    const cocclOperationDescriptor& descriptor) {
  if (info.count == 0) {
    return false;
  }
  if (info.operation == cocclOperation::SendRecv) {
    return info.func == ncclFuncSend
        ? info.sendbuff != nullptr
        : info.func == ncclFuncRecv && info.recvbuff != nullptr;
  }
  if (info.sendbuff == nullptr || info.recvbuff == nullptr) return false;
  return !cocclOperationHasTrait(
             &descriptor, cocclOperationTraitCountDivisibleByRanks) ||
      info.count % (size_t)info.comm->nRanks == 0;
}

bool totalBytes(const cocclInfo& info,
                const cocclOperationDescriptor& descriptor,
                size_t* bytes) {
  const int typeBytes = ncclTypeSize(info.datatype);
  if (typeBytes <= 0 ||
      info.count > std::numeric_limits<size_t>::max() / (size_t)typeBytes) {
    return false;
  }
  size_t value = info.count * (size_t)typeBytes;
  if (cocclOperationHasTrait(
          &descriptor, cocclOperationTraitScaleBytesByRanks)) {
    if (value > std::numeric_limits<size_t>::max() /
                    (size_t)info.comm->nRanks) {
      return false;
    }
    value *= (size_t)info.comm->nRanks;
  }
  *bytes = value;
  return true;
}

bool tunableReduction(cocclOperation operation) {
  return operation == cocclOperation::ReduceScatter ||
      operation == cocclOperation::AllReduce;
}

bool sendRecvForward(const cocclInfo& info) {
  return info.func == ncclFuncSend
      ? info.comm->rank < info.peer
      : info.comm->rank > info.peer;
}

cocclPolicyKey preparedPolicy(const cocclPreparedCall& prepared) {
  if (prepared.info.operation == cocclOperation::SendRecv &&
      prepared.trainingRole == cocclTrainingRolePipelineParallel) {
    return cocclDirectionalPolicy(cocclOperation::SendRecv,
                                  sendRecvForward(prepared.info));
  }
  return cocclDefaultPolicy(prepared.info.operation);
}

const char* compressionScopeName(cocclCompressionScope scope) {
  switch (scope) {
    case cocclCompressionScope::Default: return "default";
    case cocclCompressionScope::Intra: return "intra";
    case cocclCompressionScope::Inter: return "inter";
    case cocclCompressionScope::Count: break;
  }
  return "unknown";
}

const char* policyVariantName(cocclPolicyVariant variant) {
  switch (variant) {
    case cocclPolicyVariant::Default: return "default";
    case cocclPolicyVariant::Forward: return "forward";
    case cocclPolicyVariant::Backward: return "backward";
  }
  return "unknown";
}

ncclResult_t resolvePreparedCompressors(cocclPreparedCall* prepared) {
  prepared->policy = preparedPolicy(*prepared);
  prepared->compressors = {};
  for (cocclCompressionScope scope : {
           cocclCompressionScope::Default,
           cocclCompressionScope::Intra,
           cocclCompressionScope::Inter}) {
    cocclResolvedCompressorPolicy resolved = {};
    if (cocclResolveCompressorPolicy(
            prepared->trainingRole,
            cocclPolicyForScope(prepared->policy, scope),
            &resolved) != ncclSuccess) {
      continue;
    }
    const size_t index = static_cast<size_t>(scope);
    prepared->compressors.handles[index] = resolved.compressor;
    prepared->compressors.datatypeSupported[index] =
        compressorDatatypeSupported(
            prepared->info.datatype, resolved.compressor);
    prepared->compressors.thresholdBytes = resolved.thresholdBytes;
    INFO(COCCL_RUNTIME,
         "COCCL route comm=%p hash=%llu role=%s operation=%s policy=%s "
         "scope=%s compressor=%s",
         prepared->info.comm,
         (unsigned long long)prepared->info.comm->commHash,
         cocclTrainingRoleName(prepared->trainingRole),
         prepared->descriptor->name,
         policyVariantName(prepared->policy.variant),
         compressionScopeName(scope),
         cocclCompressorDescriptor(resolved.compressor)->name);
  }
  return prepared->compressors.anyEnabled()
      ? ncclSuccess : ncclInvalidUsage;
}

ncclResult_t prepareCall(const cocclInfo& info,
                         const cocclOperationDescriptor* descriptor,
                         cocclPreparedCall* prepared) {
  prepared->info = info;
  prepared->descriptor = descriptor;
  if (cocclTrainingAssistEnabled()) {
    cocclTrainingClassification classification;
    if (!cocclTrainingAssistQuery(info.comm, &classification) ||
        classification.role == cocclTrainingRoleUnknown) {
      return ncclInvalidUsage;
    }
    prepared->trainingRole = classification.role;
  }
  return resolvePreparedCompressors(prepared);
}

bool callSupported(const cocclInfo& info,
                   const cocclOperationDescriptor& descriptor) {
  if (info.comm->nRanks <= 1) return false;
  if (cocclOperationHasTrait(&descriptor, cocclOperationTraitReduction) &&
      info.op != ncclSum) {
    return false;
  }
  return shapeSupported(info, descriptor);
}

ncclResult_t routeNativeGroupedSendRecv(
    const cocclInfo& info, bool* isEnqueued) {
  if (ncclGroupDepth == 0 || info.operation != cocclOperation::SendRecv) {
    return ncclSuccess;
  }
  NCCLCHECK(cocclGroupEnqueueNative(&info));
  *isEnqueued = true;
  return ncclSuccess;
}

}  // namespace

ncclResult_t cocclEnqueueCheck(const cocclInfo* info, bool* isEnqueued) {
  if (isEnqueued == nullptr) return ncclInvalidArgument;
  *isEnqueued = false;
  if (info == nullptr) return ncclInvalidArgument;

  if (!cocclCompressionEnabled()) return ncclSuccess;
  if (callerGuardActive) return ncclSuccess;
  if (info->comm == nullptr) return ncclSuccess;

  cocclTrainingAssistObserve(info, ncclGroupDepth);

  const cocclOperationDescriptor* descriptor =
      cocclGetOperationDescriptor(info->operation);
  if (descriptor == nullptr || !callSupported(*info, *descriptor)) {
    return routeNativeGroupedSendRecv(*info, isEnqueued);
  }

  size_t bytes = 0;
  if (!totalBytes(*info, *descriptor, &bytes)) {
    return routeNativeGroupedSendRecv(*info, isEnqueued);
  }

  cocclPreparedCall prepared;
  if (prepareCall(*info, descriptor, &prepared) != ncclSuccess) {
    return routeNativeGroupedSendRecv(*info, isEnqueued);
  }
  if (bytes <= prepared.compressors.thresholdBytes) {
    return routeNativeGroupedSendRecv(*info, isEnqueued);
  }
  if (tunableReduction(info->operation)) {
    if (cocclSelectAlgorithm(&prepared) != ncclSuccess) {
      return routeNativeGroupedSendRecv(*info, isEnqueued);
    }
  } else if (!cocclPreparedAlgorithmSupported(
                 &prepared, cocclAlgorithmNone)) {
    return routeNativeGroupedSendRecv(*info, isEnqueued);
  }

  NCCLCHECK(cocclEnqueuePreparedCall(&prepared));
  *isEnqueued = true;
  return ncclSuccess;
}

ncclResult_t cocclEnqueueExplicitCall(
    const cocclInfo* info, cocclAlgorithmKind algorithm) {
  if (info == nullptr || info->comm == nullptr ||
      !cocclCompressionEnabled()) {
    return ncclInvalidUsage;
  }
  const cocclOperationDescriptor* descriptor =
      cocclGetOperationDescriptor(info->operation);
  if (descriptor == nullptr || !callSupported(*info, *descriptor)) {
    return ncclInvalidUsage;
  }

  cocclPreparedCall prepared;
  NCCLCHECK(prepareCall(*info, descriptor, &prepared));
  prepared.algorithm = algorithm;
  if (algorithm == cocclAlgorithmNone &&
      tunableReduction(info->operation)) {
    NCCLCHECK(cocclSelectAlgorithm(&prepared));
  }
  if (!cocclPreparedAlgorithmHasCompression(
          &prepared, prepared.algorithm)) {
    return ncclInvalidUsage;
  }
  if (!cocclPreparedAlgorithmSupported(
          &prepared, prepared.algorithm)) {
    return cocclReplayNativeCall(*info);
  }
  return cocclEnqueuePreparedCall(&prepared);
}

ncclResult_t cocclEnqueuePreparedCall(const cocclPreparedCall* prepared) {
  if (ncclGroupDepth > 0) return cocclGroupEnqueue(prepared);
  return cocclExecutePreparedCall(prepared);
}

ncclResult_t cocclReplayNativeCall(const cocclInfo& info) {
  const bool previous = callerGuardActive;
  callerGuardActive = true;

  ncclResult_t result = ncclInvalidArgument;
  switch (info.operation) {
    case cocclOperation::AllGather:
      result = ncclAllGather(info.sendbuff, info.recvbuff, info.count,
                             info.datatype, info.comm, info.stream);
      break;
    case cocclOperation::ReduceScatter:
      result = ncclReduceScatter(info.sendbuff, info.recvbuff, info.count,
                                 info.datatype, info.op, info.comm,
                                 info.stream);
      break;
    case cocclOperation::AllReduce:
      result = ncclAllReduce(info.sendbuff, info.recvbuff, info.count,
                             info.datatype, info.op, info.comm, info.stream);
      break;
    case cocclOperation::AllToAll:
      result = ncclAllToAll(info.sendbuff, info.recvbuff, info.count,
                            info.datatype, info.comm, info.stream);
      break;
    case cocclOperation::SendRecv:
      result = info.func == ncclFuncSend
          ? ncclSend(info.sendbuff, info.count, info.datatype, info.peer,
                     info.comm, info.stream)
          : ncclRecv(info.recvbuff, info.count, info.datatype, info.peer,
                     info.comm, info.stream);
      break;
    default:
      break;
  }

  callerGuardActive = previous;
  return result;
}

ncclResult_t cocclExecutePreparedCall(const cocclPreparedCall* prepared) {
  if (prepared == nullptr || prepared->descriptor == nullptr ||
      !prepared->compressors.anyEnabled()) {
    return ncclInvalidArgument;
  }

  const cocclInfo& info = prepared->info;
  const bool previous = callerGuardActive;
  callerGuardActive = true;

  ncclResult_t result = ncclInvalidArgument;
  switch (info.operation) {
    case cocclOperation::AllGather:
      result = cocclExecuteAllGather(prepared);
      break;
    case cocclOperation::AllToAll:
      result = cocclExecuteAllToAll(prepared);
      break;
    case cocclOperation::ReduceScatter:
      result = cocclExecuteReduceScatter(prepared);
      break;
    case cocclOperation::AllReduce:
      result = cocclExecuteAllReduce(prepared);
      break;
    case cocclOperation::SendRecv:
      result = cocclExecuteSendRecv(prepared);
      break;
    default:
      break;
  }

  callerGuardActive = previous;
  return result;
}
