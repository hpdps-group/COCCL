#ifndef COCCL_GROUP_INTERNAL_H_
#define COCCL_GROUP_INTERNAL_H_

#include "nccl.h"

struct cocclPreparedCall;

ncclResult_t cocclGroupEnqueue(const cocclPreparedCall* prepared);

#endif
