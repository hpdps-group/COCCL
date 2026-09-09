#include "core/backend/coccl_backend_collectives.h"

#include "core/memory/coccl_buffer_management.h"
#include "core/pipeline/coccl_pipeline.h"
#include "comm.h"
#include "collectives.h"
#include "config/collconfig.h"
#include "rma/rma.h"

namespace {

ncclCollConfig_t nativeConfig(const cocclCollectiveConfig* config) {
  ncclCollConfig_t result = NCCL_COLLCONFIG_INITIALIZER;
  if (config != nullptr) {
    result.minCTAs = config->minCTAs;
    result.maxCTAs = config->maxCTAs;
    result.CTAPolicy = config->CTAPolicy;
    result.userProfilerTag = config->userProfilerTag;
  }
  return result;
}

}  // namespace

ncclResult_t cocclBackendAllGather(
    const void* send, void* recv, size_t count, ncclDataType_t datatype,
    ncclComm_t comm, cudaStream_t stream,
    const cocclCollectiveConfig* config) {
  const ncclCollConfig_t options = nativeConfig(config);
  return ncclAllGatherConfig(
      send, recv, count, datatype, comm, stream, &options);
}

ncclResult_t cocclBackendAllToAll(
    const void* send, void* recv, size_t count, ncclDataType_t datatype,
    ncclComm_t comm, cudaStream_t stream,
    const cocclCollectiveConfig* config) {
  const ncclCollConfig_t options = nativeConfig(config);
  return ncclAlltoAllConfig(
      send, recv, count, datatype, comm, stream, &options);
}

ncclResult_t cocclBackendReduceScatter(
    const void* send, void* recv, size_t count, ncclDataType_t datatype,
    ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream,
    const cocclCollectiveConfig* config) {
  const ncclCollConfig_t options = nativeConfig(config);
  return ncclReduceScatterConfig(
      send, recv, count, datatype, op, comm, stream, &options);
}

ncclResult_t cocclBackendAllReduce(
    const void* send, void* recv, size_t count, ncclDataType_t datatype,
    ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream,
    const cocclCollectiveConfig* config) {
  const ncclCollConfig_t options = nativeConfig(config);
  return ncclAllReduceConfig(
      send, recv, count, datatype, op, comm, stream, &options);
}

void cocclBackendAllGatherCtaBounds(
    ncclComm_t comm, size_t bytes, int* minCTAs, int* maxCTAs) {
  if (comm->cudaArch == 800 && comm->nNodes == 1 &&
      bytes >= (size_t{16} << 20)) {
    *minCTAs = 9;
    *maxCTAs = 9;
  }
}

ncclResult_t cocclBackendCommSplit(
    ncclComm_t parent, int color, int key, ncclComm_t* child) {
  // The communicator policy provisions the resources used by hierarchical CE.
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.CTAPolicy = NCCL_CTA_POLICY_ZERO;
  return ncclCommSplit(parent, color, key, child, &config);
}

int cocclPipelineStageCtaPolicy(
    ncclComm_t ownerComm, const cocclPipelineStage& stage) {
  if (stage.config != nullptr &&
      stage.config->CTAPolicy != NCCL_CONFIG_UNDEF_INT) {
    return stage.config->CTAPolicy;
  }
  if (ownerComm->nNodes <= 1) return NCCL_CONFIG_UNDEF_INT;
  switch (stage.kind) {
    case cocclPipelineStageAllGather:
    case cocclPipelineStageReduceScatter:
      return NCCL_CTA_POLICY_ZERO;
    case cocclPipelineStageAllToAll:
      return NCCL_CTA_POLICY_DEFAULT;
    default:
      return NCCL_CONFIG_UNDEF_INT;
  }
}

cocclBufferRegistrationKind cocclBackendStageRegistration(
    ncclComm_t owner, const cocclPipelineStage& stage, bool framed) {
  ncclComm_t comm = stage.comm;
  const int stagePolicy = cocclPipelineStageCtaPolicy(
    owner, stage);
  const int effectivePolicy = ncclCollConfigResolveCTAPolicy(
    stagePolicy, comm->config.CTAPolicy,
    ncclGetEnvCtaPolicy() != NCCL_CONFIG_UNDEF_INT);
  const bool zeroCta =
    (effectivePolicy & NCCL_CTA_POLICY_ZERO) != 0;
  const bool framedRma = framed &&
    stage.kind == cocclPipelineStageAllToAll &&
    comm->config.rmaEagerInit && comm->hostRmaSupport &&
    comm->config.numRmaSig > 0 &&
    (comm->nNodes == 1 || ncclRmaProxyEnabled(comm));
  const bool symmetric = framedRma ||
    (!framed &&
     (stage.kind == cocclPipelineStageAllGather ||
      (stage.kind == cocclPipelineStageAllToAll && zeroCta) ||
      stage.kind == cocclPipelineStageReduceScatter));
  const cocclBufferRegistrationKind registration = framedRma
    ? cocclBufferRegistrationKind::Rma
    : (symmetric ? cocclBufferRegistrationKind::Symmetric
           : cocclBufferRegistrationKind::Ordinary);

  return registration;
}
