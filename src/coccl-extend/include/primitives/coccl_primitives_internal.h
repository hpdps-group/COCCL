#ifndef COCCL_PRIMITIVES_INTERNAL_H_
#define COCCL_PRIMITIVES_INTERNAL_H_

#include <stddef.h>
#include "comm.h"
#include "nccl.h"
#include "collectives.h"
#include "argcheck.h"
#include "enqueue.h"
#include "compression/compress.h"
#include "buffer/coccl_buffer_management.h"
#include "runtime/coccl_comm.h"
#include "runtime/coccl_runtime.h"
#include "pipeline/coccl_pipeline.h"
#include "../../graph/topo.h"

// Common implementation dependencies for primitives expressed through the
// current COCCL pipeline.

ncclResult_t cocclExecuteAllGather(const cocclPreparedCall* prepared);
ncclResult_t cocclExecuteReduceScatter(const cocclPreparedCall* prepared);
ncclResult_t cocclExecuteAllReduce(const cocclPreparedCall* prepared);
ncclResult_t cocclExecuteAllToAll(const cocclPreparedCall* prepared);
ncclResult_t cocclExecuteSendRecv(const cocclPreparedCall* prepared);

#endif
