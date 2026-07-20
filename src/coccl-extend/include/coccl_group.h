#ifndef COCCL_GROUP_H_
#define COCCL_GROUP_H_

#include "nccl.h"

struct cocclAlgorithmDecision;
struct cocclRuntimeArgs;

// COCCL collectives submitted inside an outer NCCL group are retained as
// descriptors. They are replayed only after the outermost group reaches depth
// zero, so each primitive can launch its existing pipeline normally.
bool cocclGroupHasPending();
ncclResult_t cocclGroupEnqueue(const cocclRuntimeArgs* args,
                               const cocclAlgorithmDecision* decision);
ncclResult_t cocclGroupDrain();
void cocclGroupAbort();

#endif
