#ifndef COCCL_OLD_IMPL_INTERNAL_H_
#define COCCL_OLD_IMPL_INTERNAL_H_

#include "coccl_primitives_internal.h"

static inline ncclResult_t cocclRegisterHierarchicalWorkspace(
    cocclBufferHandle* buffer, const cocclHierarchicalComms* hierarchy) {
  if (buffer == nullptr || hierarchy == nullptr) return ncclInvalidArgument;
  NCCLCHECK(cocclRegisterBufferForComm(buffer, hierarchy->intraComm));
  NCCLCHECK(cocclRegisterBufferForComm(buffer, hierarchy->interComm));
  return ncclSuccess;
}

#endif
