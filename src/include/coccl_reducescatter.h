#ifndef COCCL_REDUCESCATTER_H_
#define COCCL_REDUCESCATTER_H_

#include "nccl.h"

struct cocclPreparedCall;

ncclResult_t cocclExecuteReduceScatter(const cocclPreparedCall* prepared);

#endif
