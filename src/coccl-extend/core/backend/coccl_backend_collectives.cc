#include "core/backend/coccl_backend_collectives.h"

#include "core/memory/coccl_buffer_management.h"
#include "core/pipeline/coccl_pipeline.h"
#include "comm.h"
#include "collectives.h"

ncclResult_t cocclBackendAllGather(
    const void* send, void* recv, size_t count, ncclDataType_t datatype,
    ncclComm_t comm, cudaStream_t stream,
    const cocclCollectiveConfig*) {
  return ncclAllGather(
      send, recv, count, datatype, comm, stream);
}

ncclResult_t cocclBackendAllToAll(
    const void* send, void* recv, size_t count, ncclDataType_t datatype,
    ncclComm_t comm, cudaStream_t stream,
    const cocclCollectiveConfig*) {
  return ncclAllToAll(
      send, recv, count, datatype, comm, stream);
}

ncclResult_t cocclBackendReduceScatter(
    const void* send, void* recv, size_t count, ncclDataType_t datatype,
    ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream,
    const cocclCollectiveConfig*) {
  return ncclReduceScatter(
      send, recv, count, datatype, op, comm, stream);
}

ncclResult_t cocclBackendAllReduce(
    const void* send, void* recv, size_t count, ncclDataType_t datatype,
    ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream,
    const cocclCollectiveConfig*) {
  return ncclAllReduce(
      send, recv, count, datatype, op, comm, stream);
}

void cocclBackendAllGatherCtaBounds(
    ncclComm_t, size_t, int*, int*) {
}

ncclResult_t cocclBackendCommSplit(
    ncclComm_t parent, int color, int key, ncclComm_t* child) {
  return ncclCommSplit(parent, color, key, child, nullptr);
}

cocclBufferRegistrationKind cocclBackendStageRegistration(
    ncclComm_t, const cocclPipelineStage& stage, bool framed) {
  ncclComm_t comm = stage.comm;
  const bool symmetric = !framed && comm->nNodes == 1 &&
      (stage.kind == cocclPipelineStageAllGather ||
       stage.kind == cocclPipelineStageReduceScatter);
  return symmetric ? cocclBufferRegistrationKind::Symmetric
                   : cocclBufferRegistrationKind::Ordinary;
}
