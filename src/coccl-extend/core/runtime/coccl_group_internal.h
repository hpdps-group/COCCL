#ifndef COCCL_GROUP_INTERNAL_H_
#define COCCL_GROUP_INTERNAL_H_

#include "nccl.h"

struct cocclPreparedCall;
struct cocclInfo;

ncclResult_t cocclGroupEnqueue(const cocclPreparedCall* prepared);
ncclResult_t cocclGroupEnqueueNative(const cocclInfo* info);

#endif
