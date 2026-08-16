#ifndef COCCL_ALLREDUCE_H_
#define COCCL_ALLREDUCE_H_

#include "nccl.h"

struct cocclPreparedCall;

ncclResult_t cocclExecuteAllReduce(const cocclPreparedCall* prepared);

#endif
