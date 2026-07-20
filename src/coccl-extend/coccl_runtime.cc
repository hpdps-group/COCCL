#include "coccl_runtime.h"

#include "coccl_autotune.h"
#include "coccl_comm.h"
#include "coccl_group.h"
#include "coccl_training_assist.h"
#include "collectives.h"
#include "comm.h"
#include "group.h"
#include "info.h"
#include "strongstream.h"

#include <limits>

namespace {

__thread bool cocclCallerGuardActive = false;

bool cocclRuntimeCommReady(ncclComm_t comm) {
  return comm != nullptr && cocclCommAvailable(comm);
}

bool cocclResolveCompressorOp(
    const cocclRuntimeArgs* args, ncclCommOp_t* compressorOp) {
  if (args == nullptr || args->comm == nullptr || compressorOp == nullptr) {
    return false;
  }

  switch (args->func) {
    case ncclFuncAllGather:
      *compressorOp = AllGather;
      return true;
    case ncclFuncAllReduce:
      *compressorOp = AllReduce;
      return true;
    case ncclFuncReduceScatter:
      *compressorOp = ReduceScatter;
      return true;
    case ncclFuncSend:
      *compressorOp = args->comm->rank < args->peer
          ? SendRecv : SendRecv_BWD;
      return true;
    case ncclFuncRecv:
      *compressorOp = args->comm->rank > args->peer
          ? SendRecv : SendRecv_BWD;
      return true;
    default:
      return false;
  }
}

bool cocclComputeTotalBytes(
    const cocclRuntimeArgs* args, size_t* totalBytes) {
  if (args == nullptr || args->comm == nullptr || totalBytes == nullptr) {
    return false;
  }

  const int datatypeBytes = ncclTypeSize(args->datatype);
  if (datatypeBytes <= 0 ||
      args->count > std::numeric_limits<size_t>::max() /
                        (size_t)datatypeBytes) {
    return false;
  }

  size_t bytes = args->count * (size_t)datatypeBytes;
  if (args->func == ncclFuncAllGather ||
      args->func == ncclFuncReduceScatter) {
    if (args->comm->nRanks <= 0 ||
        bytes > std::numeric_limits<size_t>::max() /
                    (size_t)args->comm->nRanks) {
      return false;
    }
    bytes *= (size_t)args->comm->nRanks;
  }
  *totalBytes = bytes;
  return true;
}

bool cocclResolveTrainingRole(
    ncclComm_t comm, cocclTrainingRole* role) {
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

bool cocclCanRoute(const cocclRuntimeArgs* args) {
  ncclCommOp_t compressorOp;
  size_t totalBytes = 0;
  cocclTrainingRole role = cocclTrainingRoleUnknown;
  if (!cocclResolveCompressorOp(args, &compressorOp) ||
      !cocclComputeTotalBytes(args, &totalBytes) ||
      !cocclResolveTrainingRole(args->comm, &role)) {
    return false;
  }

  // Algorithm selection runs only after this generic compression predicate.
  // Internal Inter/BWD compressor tags share their base primitive threshold.
  return cocclCommShouldCompress(
      args->comm, compressorOp, role, totalBytes, args->datatype, args->op);
}

bool cocclGroupRouteSupported(const cocclRuntimeArgs* args) {
  if (ncclGroupDepth == 0) return true;

  // Deferred replay currently supports complete collective primitives only.
  // P2P has a separate split-communicator lifecycle and remains native inside
  // an outer group.
  if (args->func == ncclFuncSend || args->func == ncclFuncRecv) return false;
  if (!args->comm->config.blocking) return false;

  // Pipeline-owned streams and events are not capture aware. Let NCCL's native
  // path preserve CUDA Graph semantics.
  ncclCudaGraph graph = ncclCudaGraphNone();
  if (ncclCudaGetCapturingGraph(&graph, args->stream) != ncclSuccess) {
    return false;
  }
  return !ncclCudaGraphValid(graph);
}

}  // namespace

bool cocclAvailable(const cocclRuntimeArgs* args) {
  if (args == nullptr || !cocclCompressionEnabled() ||
      !cocclRuntimeCommReady(args->comm) || cocclCallerGuardActive) {
    return false;
  }

  // Observe the upper-layer call before operation flags, datatype restrictions,
  // thresholds, or group fallback filter it. Internal calls are excluded by
  // the caller guard, and deferred replay bypasses this entry point.
  cocclTrainingAssistObserve(args, ncclGroupDepth);
  return cocclCanRoute(args) && cocclGroupRouteSupported(args);
}

ncclResult_t cocclExecutePrimitive(
    const cocclRuntimeArgs* args, const cocclAlgorithmDecision* decision) {
  if (args == nullptr || args->comm == nullptr || decision == nullptr) {
    return ncclInvalidArgument;
  }

  ncclResult_t ret = ncclSuccess;
  bool previousCallerFlag = cocclCallerGuardActive;
  // COCCL primitives call NCCL collectives internally. Guard those nested calls
  // from being routed recursively back into COCCL.
  cocclCallerGuardActive = true;
  switch (args->func) {
    case ncclFuncAllGather:
      NCCLCHECKGOTO(
          ncclAllGatherCompOverlap(
              args->sendbuff, args->recvbuff, args->count, args->datatype,
              args->comm, args->stream),
          ret, exit);
      break;
    case ncclFuncReduceScatter:
      if (decision->algorithm == cocclAlgorithmReduceScatterOneShot) {
        NCCLCHECKGOTO(
            ncclReduceScatterCompOneShotOverlap(
                args->sendbuff, args->recvbuff, args->count, args->datatype,
                args->op, args->comm, args->stream),
            ret, exit);
      } else if (decision->algorithm == cocclAlgorithmReduceScatterTwoShot) {
        NCCLCHECKGOTO(
            ncclReduceScatterCompTwoShotTLOverlap(
                args->sendbuff, args->recvbuff, args->count, args->datatype,
                args->op, args->comm, args->stream),
            ret, exit);
      } else {
        ret = ncclInternalError;
      }
      break;
    case ncclFuncAllReduce:
      if (decision->algorithm == cocclAlgorithmAllReduceOneShot) {
        NCCLCHECKGOTO(
            ncclAllReduceCompOneShot(
                args->sendbuff, args->recvbuff, args->count, args->datatype,
                args->op, args->comm, args->stream),
            ret, exit);
      } else if (decision->algorithm == cocclAlgorithmAllReduceTwoShot) {
        NCCLCHECKGOTO(
            ncclAllReduceCompTwoShotOverlap(
                args->sendbuff, args->recvbuff, args->count, args->datatype,
                args->op, args->comm, args->stream),
            ret, exit);
      } else if (decision->algorithm == cocclAlgorithmAllReduceTripleShot) {
        NCCLCHECKGOTO(
            ncclAllReduceCompTripleShotTLOverlap(
                args->sendbuff, args->recvbuff, args->count, args->datatype,
                args->op, args->comm, args->stream),
            ret, exit);
      } else {
        ret = ncclInternalError;
      }
      break;
    case ncclFuncSend:
      NCCLCHECKGOTO(
          ncclSendComp(args->sendbuff, args->count, args->datatype, args->peer,
                       args->comm, args->stream),
          ret, exit);
      break;
    case ncclFuncRecv:
      NCCLCHECKGOTO(
          ncclRecvDecomp(args->recvbuff, args->count, args->datatype,
                         args->peer, args->comm, args->stream),
          ret, exit);
      break;
    default:
      ret = ncclInvalidArgument;
      break;
  }

exit:
  cocclCallerGuardActive = previousCallerFlag;
  return ret;
}

ncclResult_t cocclEnqueueCheck(const cocclRuntimeArgs* args) {
  if (args == nullptr || args->comm == nullptr) return ncclInvalidArgument;

  cocclAlgorithmDecision decision = {};
  ncclResult_t ret = ncclSuccess;
  if (args->func == ncclFuncReduceScatter || args->func == ncclFuncAllReduce) {
    ret = cocclSelectAlgorithm(args, &decision);
    if (ret != ncclSuccess) {
      return ncclGroupDepth > 0 ? ncclGroupErrCheck(ret) : ret;
    }
  }

  if (ncclGroupDepth > 0) {
    if (args->func != ncclFuncAllGather &&
        args->func != ncclFuncReduceScatter &&
        args->func != ncclFuncAllReduce) {
      return ncclGroupErrCheck(ncclInvalidUsage);
    }
    ret = cocclGroupEnqueue(args, &decision);
    return ret == ncclSuccess ? ret : ncclGroupErrCheck(ret);
  }

  return cocclExecutePrimitive(args, &decision);
}
