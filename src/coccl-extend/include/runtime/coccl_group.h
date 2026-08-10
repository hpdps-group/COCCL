#ifndef COCCL_GROUP_H_
#define COCCL_GROUP_H_

#include "nccl.h"

struct cocclPreparedCall;

// COCCL collectives submitted inside an outer NCCL group are retained as
// prepared calls. They are replayed only after the outermost group reaches
// depth zero. The contract is one host thread per CUDA rank; enqueue rejects a
// batch spanning multiple local devices on one thread.
bool cocclGroupHasPending();
ncclResult_t cocclGroupEnqueue(const cocclPreparedCall* prepared);
ncclResult_t cocclGroupDrain();
void cocclGroupAbort();

#endif
