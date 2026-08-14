#ifndef COCCL_PRIMITIVE_DISPATCH_H_
#define COCCL_PRIMITIVE_DISPATCH_H_

#include "nccl.h"

#include <stddef.h>

struct cocclPreparedCall;

ncclResult_t cocclExecuteAllGather(const cocclPreparedCall* prepared);
ncclResult_t cocclExecuteReduceScatter(const cocclPreparedCall* prepared);
ncclResult_t cocclExecuteAllReduce(const cocclPreparedCall* prepared);
ncclResult_t cocclExecuteAllToAll(const cocclPreparedCall* prepared);
ncclResult_t cocclExecuteSendRecv(const cocclPreparedCall* prepared);
ncclResult_t cocclExecuteSendRecvBatch(
    const cocclPreparedCall* calls, size_t count);

#endif
