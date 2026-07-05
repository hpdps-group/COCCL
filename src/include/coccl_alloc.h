#ifndef COCCL_ALLOC_H_
#define COCCL_ALLOC_H_

#include "nccl.h"
#include "argcheck.h"

ncclResult_t cocclBuffAlloc(void** buff, size_t expectBytes, ncclComm_t comm);
#endif