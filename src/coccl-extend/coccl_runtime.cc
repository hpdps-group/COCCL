#include "runtime/coccl_runtime.h"

#include "primitives/coccl_primitives_internal.h"
#include "runtime/coccl_autotune.h"
#include "runtime/coccl_comm.h"
#include "runtime/coccl_group.h"
#include "training/coccl_training_assist.h"
#include "comm.h"
#include "group.h"
#include "strongstream.h"

#include <limits>
#include <utility>

namespace {

__thread bool cocclCallerGuardActive = false;

bool cocclComputeTotalBytes(const cocclInfo& info,
                            const cocclOperationDescriptor& descriptor,
                            size_t* totalBytes) {
  if (info.comm == nullptr || totalBytes == nullptr) return false;
  const int datatypeBytes = ncclTypeSize(info.datatype);
  if (datatypeBytes <= 0 ||
      info.count > std::numeric_limits<size_t>::max() /
                       (size_t)datatypeBytes) {
    return false;
  }

  size_t bytes = info.count * (size_t)datatypeBytes;
  if (cocclOperationHasTrait(
          &descriptor, cocclOperationTraitScaleBytesByRanks)) {
    if (info.comm->nRanks <= 0 ||
        bytes > std::numeric_limits<size_t>::max() /
                    (size_t)info.comm->nRanks) {
      return false;
    }
    bytes *= (size_t)info.comm->nRanks;
  }
  *totalBytes = bytes;
  return true;
}

bool cocclResolveTrainingRole(ncclComm_t comm, cocclTrainingRole* role) {
  if (role == nullptr) return false;
  *role = cocclTrainingRoleUnknown;
  if (!cocclTrainingAssistEnabled()) return true;

  cocclTrainingClassification classification;
  if (!cocclTrainingAssistQuery(comm, &classification) ||
      classification.role == cocclTrainingRoleUnknown) {
    return false;
  }
  *role = classification.role;
  return true;
}

bool cocclCompressionDatatypeSupported(ncclDataType_t datatype) {
  if (datatype == ncclFloat16 || datatype == ncclFloat32) return true;
#if defined(__CUDA_BF16_TYPES_EXIST__)
  return datatype == ncclBfloat16;
#else
  return false;
#endif
}

bool cocclSendRecvForward(const cocclInfo& info) {
  return info.func == ncclFuncSend
      ? info.comm->rank < info.peer
      : info.comm->rank > info.peer;
}

cocclPolicyKey cocclResolvePolicyKey(const cocclInfo& info,
                                     cocclTrainingRole role) {
  if (info.operation == cocclOperation::SendRecv &&
      role == cocclTrainingRolePipelineParallel) {
    return cocclDirectionalPolicy(info.operation,
                                  cocclSendRecvForward(info));
  }
  return cocclDefaultPolicy(info.operation);
}

bool cocclGroupRouteSupported(const cocclInfo& info,
                              const cocclOperationDescriptor& descriptor) {
  if (ncclGroupDepth == 0) return true;
  if (!cocclOperationHasTrait(&descriptor, cocclOperationTraitGrouped) ||
      !info.comm->config.blocking) {
    return false;
  }

  // Pipeline streams and events are not capture aware. Native NCCL keeps CUDA
  // Graph behavior for grouped calls under capture.
  ncclCudaGraph graph = ncclCudaGraphNone();
  if (ncclCudaGetCapturingGraph(&graph, info.stream) != ncclSuccess) {
    return false;
  }
  return !ncclCudaGraphValid(graph);
}

bool cocclShapeSupported(const cocclInfo& info,
                         const cocclOperationDescriptor& descriptor) {
  return !cocclOperationHasTrait(
             &descriptor, cocclOperationTraitCountDivisibleByRanks) ||
      (info.comm->nRanks > 0 &&
       info.count % (size_t)info.comm->nRanks == 0);
}

}  // namespace

ncclResult_t cocclEnqueueCheck(const cocclInfo* info, bool* isEnqueued) {
  if (isEnqueued == nullptr) return ncclInvalidArgument;
  *isEnqueued = false;
  if (info == nullptr) return ncclInvalidArgument;

  if (!cocclCompressionEnabled()) return ncclSuccess;
  if (cocclCallerGuardActive) return ncclSuccess;
  if (info->comm == nullptr) return ncclSuccess;

  // Observe the public call before datatype, threshold, shape, or group
  // routing filters it. Nested primitive calls are hidden by the caller guard.
  cocclTrainingAssistObserve(info, ncclGroupDepth);

  const cocclOperationDescriptor* descriptor =
      cocclGetOperationDescriptor(info->operation);
  if (descriptor == nullptr) return ncclSuccess;
  if (info->comm->nRanks <= 1) return ncclSuccess;
  if (!cocclCompressionDatatypeSupported(info->datatype)) return ncclSuccess;
  if (cocclOperationHasTrait(descriptor, cocclOperationTraitReduction) &&
      info->op != ncclSum) {
    return ncclSuccess;
  }
  if (!cocclShapeSupported(*info, *descriptor)) return ncclSuccess;
  if (!cocclGroupRouteSupported(*info, *descriptor)) return ncclSuccess;

  size_t totalBytes = 0;
  if (!cocclComputeTotalBytes(*info, *descriptor, &totalBytes)) {
    return ncclSuccess;
  }
  cocclTrainingRole role = cocclTrainingRoleUnknown;
  if (!cocclResolveTrainingRole(info->comm, &role)) return ncclSuccess;

  const cocclPolicyKey policy = cocclResolvePolicyKey(*info, role);
  if (!cocclOperationSupportsPolicy(descriptor, policy.variant)) {
    return ncclSuccess;
  }
  cocclResolvedCompressorPolicy resolved;
  if (cocclCommResolveCompressorPolicy(
          info->comm, role, policy, &resolved) != ncclSuccess) {
    return ncclSuccess;
  }
  if (totalBytes <= resolved.thresholdBytes) {
    return ncclSuccess;
  }

  cocclPreparedCall prepared;
  prepared.info = *info;
  prepared.descriptor = descriptor;
  prepared.trainingRole = role;
  prepared.policy = policy;
  prepared.compressor = std::move(resolved.compressor);
  if (info->operation == cocclOperation::ReduceScatter ||
      info->operation == cocclOperation::AllReduce) {
    ncclResult_t ret = cocclSelectAlgorithm(&prepared);
    if (ret != ncclSuccess) {
      return ncclGroupDepth > 0 ? ncclGroupErrCheck(ret) : ret;
    }
  }

  NCCLCHECK(cocclEnqueuePreparedCall(&prepared));
  *isEnqueued = true;
  return ncclSuccess;
}

ncclResult_t cocclEnqueuePreparedCall(const cocclPreparedCall* prepared) {
  if (prepared == nullptr || prepared->info.comm == nullptr ||
      prepared->descriptor == nullptr || !prepared->compressor) {
    return ncclInvalidArgument;
  }
  if (ncclGroupDepth > 0) {
    ncclResult_t ret = cocclGroupEnqueue(prepared);
    return ret == ncclSuccess ? ret : ncclGroupErrCheck(ret);
  }
  return cocclExecutePreparedCall(prepared);
}

ncclResult_t cocclExecutePreparedCall(const cocclPreparedCall* prepared) {
  if (prepared == nullptr || prepared->info.comm == nullptr ||
      prepared->descriptor == nullptr || !prepared->compressor ||
      prepared->descriptor->operation != prepared->info.operation) {
    return ncclInvalidArgument;
  }

  const bool previousCallerFlag = cocclCallerGuardActive;
  cocclCallerGuardActive = true;
  ncclResult_t ret = ncclInvalidArgument;
  switch (prepared->info.operation) {
    case cocclOperation::AllGather:
      ret = cocclExecuteAllGather(prepared);
      break;
    case cocclOperation::ReduceScatter:
      ret = cocclExecuteReduceScatter(prepared);
      break;
    case cocclOperation::AllReduce:
      ret = cocclExecuteAllReduce(prepared);
      break;
    case cocclOperation::AllToAll:
      ret = cocclExecuteAllToAll(prepared);
      break;
    case cocclOperation::SendRecv:
      ret = cocclExecuteSendRecv(prepared);
      break;
    default:
      break;
  }
  cocclCallerGuardActive = previousCallerFlag;
  return ret;
}
