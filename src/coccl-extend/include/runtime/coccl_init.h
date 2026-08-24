#ifndef COCCL_INIT_H_
#define COCCL_INIT_H_

#include "nccl.h"

ncclResult_t cocclInit(ncclComm_t comm);
ncclResult_t cocclDestroy(ncclComm_t comm);

#endif
