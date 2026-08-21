#ifndef COCCL_GROUP_H_
#define COCCL_GROUP_H_

#include "nccl.h"

bool cocclGroupHasPending();
ncclResult_t cocclGroupPrepareEnd(bool nativePending);
ncclResult_t cocclGroupDrain();
void cocclGroupAbort();

#endif
