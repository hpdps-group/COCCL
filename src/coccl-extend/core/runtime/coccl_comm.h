#ifndef COCCL_COMM_H_
#define COCCL_COMM_H_

#include "nccl.h"

struct cocclHierarchicalComms {
  ncclComm_t ownerComm = nullptr;
  ncclComm_t intraComm = nullptr;
  ncclComm_t interComm = nullptr;
};

ncclResult_t cocclCommCreate(ncclComm_t comm);
ncclResult_t cocclCommDestroy(ncclComm_t comm);
ncclResult_t cocclCommGetZeroCtaComm(
    ncclComm_t comm, ncclComm_t* zeroCtaComm);
ncclResult_t cocclCommGetHierarchicalComms(
    ncclComm_t comm, cocclHierarchicalComms* hierarchy);

#endif
