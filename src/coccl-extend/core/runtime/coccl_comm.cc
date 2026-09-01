#include "core/runtime/coccl_comm.h"

#include "checks.h"
#include "comm.h"
#include "core/training/coccl_training_assist.h"
#include "core/tuning/coccl_autotune_internal.h"

#include <map>
#include <pthread.h>

namespace {

struct cocclHierarchicalCommState {
  ncclComm_t intraComm = nullptr;
  ncclComm_t interComm = nullptr;
};

pthread_mutex_t commLock = PTHREAD_MUTEX_INITIALIZER;
std::map<ncclComm_t, cocclHierarchicalCommState> comms;

ncclResult_t destroyHierarchy(cocclHierarchicalCommState hierarchy) {
  ncclResult_t result = ncclSuccess;
  if (hierarchy.intraComm != nullptr) {
    const ncclResult_t destroyResult = ncclCommDestroy(hierarchy.intraComm);
    if (result == ncclSuccess) result = destroyResult;
  }
  if (hierarchy.interComm != nullptr) {
    const ncclResult_t destroyResult = ncclCommDestroy(hierarchy.interComm);
    if (result == ncclSuccess) result = destroyResult;
  }
  return result;
}

}  // namespace

ncclResult_t cocclCommCreate(ncclComm_t comm) {
  pthread_mutex_lock(&commLock);
  comms.emplace(comm, cocclHierarchicalCommState{});
  pthread_mutex_unlock(&commLock);
  return ncclSuccess;
}

ncclResult_t cocclCommDestroy(ncclComm_t comm) {
  cocclAutotuneTopologyCommDestroy(comm);
  cocclHierarchicalCommState hierarchy;
  pthread_mutex_lock(&commLock);
  auto found = comms.find(comm);
  if (found != comms.end()) {
    hierarchy = found->second;
    comms.erase(found);
  }
  pthread_mutex_unlock(&commLock);
  return destroyHierarchy(hierarchy);
}

ncclResult_t cocclCommGetHierarchicalComms(
    ncclComm_t comm, cocclHierarchicalComms* hierarchy) {
  pthread_mutex_lock(&commLock);
  cocclHierarchicalCommState& state = comms.at(comm);
  if (state.intraComm != nullptr) {
    *hierarchy = {comm, state.intraComm, state.interComm};
    pthread_mutex_unlock(&commLock);
    return ncclSuccess;
  }
  pthread_mutex_unlock(&commLock);

  cocclHierarchicalCommState created;
  const int localRanks = comm->localRanks;
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  ncclResult_t result = ncclCommSplit(
      comm, comm->rank / localRanks, comm->rank, &created.intraComm, nullptr);
  if (result == ncclSuccess) {
    cocclTrainingAssistUnregister(created.intraComm);
    result = ncclCommSplit(
        comm, comm->rank % localRanks, comm->rank, &created.interComm,
        nullptr);
  }
  if (result != ncclSuccess) {
    (void)destroyHierarchy(created);
    return result;
  }
  cocclTrainingAssistUnregister(created.interComm);

  pthread_mutex_lock(&commLock);
  comms.at(comm) = created;
  pthread_mutex_unlock(&commLock);
  *hierarchy = {comm, created.intraComm, created.interComm};
  return ncclSuccess;
}
