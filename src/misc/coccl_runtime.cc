#include "coccl_runtime.h"

#include "coccl_config.h"
#include "coccl_alltoall.h"
#include "coccl_group_internal.h"
#include "coccl_prepared_call.h"
#include "collectives.h"
#include "comm.h"
#include "compress.h"
#include "group.h"

#include <limits>

namespace {

__thread bool callerGuardActive = false;

bool datatypeSupported(ncclDataType_t datatype) {
  if (datatype == ncclFloat16 || datatype == ncclFloat32) return true;
#if defined(__CUDA_BF16_TYPES_EXIST__)
  return datatype == ncclBfloat16;
#else
  return false;
#endif
}

bool shapeSupported(const cocclInfo& info,
                    const cocclOperationDescriptor& descriptor) {
  if (info.count == 0 || info.sendbuff == nullptr || info.recvbuff == nullptr) {
    return false;
  }
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

bool hierarchicalTopology(ncclComm_t comm) {
  return comm->nNodes > 1 && comm->localRanks > 1 &&
      comm->nRanks == comm->nNodes * comm->localRanks;
}

cocclAlgorithmKind selectAlgorithm(const cocclInfo& info) {
  if (info.operation != cocclOperation::ReduceScatter &&
      info.operation != cocclOperation::AllReduce) {
    return cocclAlgorithmNone;
  }
  const cocclAutotuneConfig& config = cocclGetConfig().autotune;
  if (info.operation == cocclOperation::ReduceScatter) {
    if (config.reduceScatterAlgorithm ==
        cocclReduceScatterAlgorithmPolicy::OneShot) {
      return cocclAlgorithmReduceScatterOneShot;
    }
    if (config.reduceScatterAlgorithm ==
            cocclReduceScatterAlgorithmPolicy::TwoShot &&
        !hierarchicalTopology(info.comm)) {
      return cocclAlgorithmReduceScatterOneShot;
    }
    return hierarchicalTopology(info.comm)
        ? cocclAlgorithmReduceScatterTwoShot
        : cocclAlgorithmReduceScatterOneShot;
  }
  if (info.operation == cocclOperation::AllReduce) {
    switch (config.allReduceAlgorithm) {
      case cocclAllReduceAlgorithmPolicy::OneShot:
        return cocclAlgorithmAllReduceOneShot;
      case cocclAllReduceAlgorithmPolicy::TripleShot:
        return hierarchicalTopology(info.comm)
            ? cocclAlgorithmAllReduceTripleShot
            : cocclAlgorithmAllReduceTwoShot;
      case cocclAllReduceAlgorithmPolicy::Auto:
      case cocclAllReduceAlgorithmPolicy::TwoShot:
        return cocclAlgorithmAllReduceTwoShot;
    }
  }
  return cocclAlgorithmNone;
}

bool hierarchicalAlgorithm(cocclAlgorithmKind algorithm) {
  return algorithm == cocclAlgorithmReduceScatterTwoShot ||
      algorithm == cocclAlgorithmAllReduceTripleShot;
}

ncclResult_t resolvePreparedCompressor(
    cocclPreparedCall* prepared, cocclResolvedCompressorPolicy* resolved) {
  prepared->policy = hierarchicalAlgorithm(prepared->algorithm)
      ? cocclHierarchicalPolicy(prepared->info.operation)
      : cocclDefaultPolicy(prepared->info.operation);
  NCCLCHECK(cocclResolveCompressorPolicy(prepared->policy, resolved));
  prepared->compressor = resolved->compressor;
  return ncclSuccess;
}

bool callSupported(const cocclInfo& info,
                   const cocclOperationDescriptor& descriptor) {
  if (info.comm->nRanks <= 1 || !datatypeSupported(info.datatype)) {
    return false;
  }
  if (cocclOperationHasTrait(&descriptor, cocclOperationTraitReduction) &&
      info.op != ncclSum) {
    return false;
  }
  return shapeSupported(info, descriptor);
}

}  // namespace

ncclResult_t cocclEnqueueCheck(const cocclInfo* info, bool* isEnqueued) {
  if (isEnqueued == nullptr) return ncclInvalidArgument;
  *isEnqueued = false;
  if (info == nullptr) return ncclInvalidArgument;

  if (!cocclCompressionEnabled()) return ncclSuccess;
  if (callerGuardActive) return ncclSuccess;
  if (info->comm == nullptr) return ncclSuccess;

  const cocclOperationDescriptor* descriptor =
      cocclGetOperationDescriptor(info->operation);
  if (descriptor == nullptr || !callSupported(*info, *descriptor)) {
    return ncclSuccess;
  }

  size_t bytes = 0;
  if (!totalBytes(*info, *descriptor, &bytes)) return ncclSuccess;

  cocclPreparedCall prepared;
  prepared.info = *info;
  prepared.descriptor = descriptor;
  prepared.algorithm = selectAlgorithm(*info);
  cocclResolvedCompressorPolicy resolved;
  if (resolvePreparedCompressor(&prepared, &resolved) != ncclSuccess) {
    return ncclSuccess;
  }
  if (bytes <= resolved.thresholdBytes) return ncclSuccess;

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
  prepared.info = *info;
  prepared.descriptor = descriptor;
  prepared.algorithm = algorithm == cocclAlgorithmNone
      ? selectAlgorithm(*info) : algorithm;
  cocclResolvedCompressorPolicy resolved;
  NCCLCHECK(resolvePreparedCompressor(&prepared, &resolved));
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
    default:
      break;
  }

  callerGuardActive = previous;
  return result;
}

ncclResult_t cocclExecutePreparedCall(const cocclPreparedCall* prepared) {
  if (prepared == nullptr || prepared->descriptor == nullptr ||
      prepared->compressor == nullptr) {
    return ncclInvalidArgument;
  }

  const cocclInfo& info = prepared->info;
  const bool previous = callerGuardActive;
  callerGuardActive = true;

  ncclResult_t result = ncclInvalidArgument;
  switch (info.operation) {
    case cocclOperation::AllGather:
      result = ncclAllGatherCompOverlap(
          info.sendbuff, info.recvbuff, info.count, info.datatype,
          info.comm, info.stream);
      break;
    case cocclOperation::AllToAll:
      result = cocclExecuteAllToAll(prepared);
      break;
    case cocclOperation::ReduceScatter:
      result = prepared->algorithm == cocclAlgorithmReduceScatterTwoShot
          ? ncclReduceScatterCompTwoShotOverlap(
                info.sendbuff, info.recvbuff, info.count, info.datatype,
                info.op, info.comm, info.stream)
          : ncclReduceScatterCompOneShotOverlap(
                info.sendbuff, info.recvbuff, info.count, info.datatype,
                info.op, info.comm, info.stream);
      break;
    case cocclOperation::AllReduce:
      if (prepared->algorithm == cocclAlgorithmAllReduceOneShot) {
        result = ncclAllReduceCompOneShot(
            info.sendbuff, info.recvbuff, info.count, info.datatype,
            info.op, info.comm, info.stream);
      } else if (prepared->algorithm ==
                 cocclAlgorithmAllReduceTripleShot) {
        result = ncclAllReduceCompTripleShotTLOverlap(
            info.sendbuff, info.recvbuff, info.count, info.datatype,
            info.op, info.comm, info.stream);
      } else {
        result = ncclAllReduceCompTwoShotOverlap(
            info.sendbuff, info.recvbuff, info.count, info.datatype,
            info.op, info.comm, info.stream);
      }
      break;
    default:
      break;
  }

  callerGuardActive = previous;
  return result;
}
