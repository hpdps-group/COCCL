#ifndef COCCL_GROUP_H_
#define COCCL_GROUP_H_

#include "nccl.h"

// Calls submitted inside an outer NCCL group are retained until GroupEnd.
// Prepared collectives are replayed after the native group closes; a pure
// Send/Recv batch is executed as one metadata phase and one payload phase.
bool cocclGroupHasPending();
ncclResult_t cocclGroupPrepareEnd(bool nativePending);
ncclResult_t cocclGroupDrain();
void cocclGroupAbort();

#endif
