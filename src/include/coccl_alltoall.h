#ifndef COCCL_ALLTOALL_H_
#define COCCL_ALLTOALL_H_

#include "nccl.h"

struct cocclPreparedCall;

ncclResult_t cocclExecuteAllToAll(const cocclPreparedCall* prepared);

#endif
