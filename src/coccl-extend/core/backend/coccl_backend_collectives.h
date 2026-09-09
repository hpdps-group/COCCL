#ifndef COCCL_BACKEND_COLLECTIVES_H_
#define COCCL_BACKEND_COLLECTIVES_H_

#include "nccl.h"

#include <stddef.h>
#include <stdint.h>

struct cocclPipelineStage;
enum class cocclBufferRegistrationKind;

// Only the controls used by COCCL's internal stages, not the public NCCL config.
struct cocclCollectiveConfig {
  int minCTAs = NCCL_CONFIG_UNDEF_INT;
  int maxCTAs = NCCL_CONFIG_UNDEF_INT;
  int CTAPolicy = NCCL_CONFIG_UNDEF_INT;
  uint64_t userProfilerTag = 0;
};

ncclResult_t cocclBackendAllGather(
    const void* send, void* recv, size_t count, ncclDataType_t datatype,
    ncclComm_t comm, cudaStream_t stream,
    const cocclCollectiveConfig* config = nullptr);
ncclResult_t cocclBackendAllToAll(
    const void* send, void* recv, size_t count, ncclDataType_t datatype,
    ncclComm_t comm, cudaStream_t stream,
    const cocclCollectiveConfig* config = nullptr);
ncclResult_t cocclBackendReduceScatter(
    const void* send, void* recv, size_t count, ncclDataType_t datatype,
    ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream,
    const cocclCollectiveConfig* config = nullptr);
ncclResult_t cocclBackendAllReduce(
    const void* send, void* recv, size_t count, ncclDataType_t datatype,
    ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream,
    const cocclCollectiveConfig* config = nullptr);
void cocclBackendAllGatherCtaBounds(
    ncclComm_t comm, size_t bytes, int* minCTAs, int* maxCTAs);
ncclResult_t cocclBackendCommSplit(
    ncclComm_t parent, int color, int key, ncclComm_t* child);
cocclBufferRegistrationKind cocclBackendStageRegistration(
    ncclComm_t owner, const cocclPipelineStage& stage, bool framed);

#endif
