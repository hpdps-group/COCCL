#ifndef COCCL_PRIMITIVE_DISPATCH_H_
#define COCCL_PRIMITIVE_DISPATCH_H_

#include "nccl.h"
#include "core/runtime/coccl_prepared_call.h"

#include <stddef.h>

struct cocclPipelineSpec;
struct cocclPipelineStage;

cocclPipelineSpec cocclBuildAllGatherSpec(
    const cocclInfo& info, cocclAlgorithmKind algorithm, void* compressor,
    ncclComm_t gatherComm, ncclComm_t intraComm,
    cocclPipelineStage* stages);

ncclResult_t cocclExecuteAllGather(const cocclPreparedCall* prepared);
ncclResult_t cocclExecuteAllToAll(const cocclPreparedCall* prepared);
ncclResult_t cocclExecuteReduceScatter(const cocclPreparedCall* prepared);
ncclResult_t cocclExecuteAllReduce(const cocclPreparedCall* prepared);
ncclResult_t cocclExecuteSendRecv(const cocclPreparedCall* prepared);
ncclResult_t cocclExecuteSendRecvBatch(
    const cocclPreparedCall* calls, size_t count);

#endif
