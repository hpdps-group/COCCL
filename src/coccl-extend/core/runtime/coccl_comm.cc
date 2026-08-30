#include "core/runtime/coccl_comm.h"

#include "checks.h"
#include "comm.h"
#include "core/training/coccl_training_assist.h"

#include <map>
#include <pthread.h>

namespace {

struct cocclCommState {
  ncclComm_t zeroCtaComm = nullptr;
  ncclComm_t intraComm = nullptr;
  ncclComm_t interComm = nullptr;
};

pthread_mutex_t commLock = PTHREAD_MUTEX_INITIALIZER;
std::map<ncclComm_t, cocclCommState> comms;

ncclResult_t destroyComms(cocclCommState state) {
  ncclResult_t result = ncclSuccess;
  if (state.zeroCtaComm != nullptr) {
    const ncclResult_t destroyResult =
        ncclCommDestroy(state.zeroCtaComm);
    if (result == ncclSuccess) result = destroyResult;
  }
  if (state.intraComm != nullptr) {
    const ncclResult_t destroyResult = ncclCommDestroy(state.intraComm);
    if (result == ncclSuccess) result = destroyResult;
  }
  if (state.interComm != nullptr) {
    const ncclResult_t destroyResult = ncclCommDestroy(state.interComm);
    if (result == ncclSuccess) result = destroyResult;
  }
  return result;
}

ncclResult_t splitZeroCtaComm(
    ncclComm_t parent, int color, int key, ncclComm_t* child) {
  // Hierarchical CE provisions its internal RMA contexts from the
  // communicator policy; the per-call policy only selects a prepared stage.
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.CTAPolicy = NCCL_CTA_POLICY_ZERO;
  NCCLCHECK(ncclCommSplit(parent, color, key, child, &config));
  cocclTrainingAssistUnregister(*child);
  return ncclSuccess;
}

}  // namespace

ncclResult_t cocclCommCreate(ncclComm_t comm) {
  pthread_mutex_lock(&commLock);
  comms.emplace(comm, cocclCommState{});
  pthread_mutex_unlock(&commLock);
  return ncclSuccess;
}

ncclResult_t cocclCommDestroy(ncclComm_t comm) {
  cocclCommState state;
  pthread_mutex_lock(&commLock);
  auto found = comms.find(comm);
  if (found != comms.end()) {
    state = found->second;
    comms.erase(found);
  }
  pthread_mutex_unlock(&commLock);
  return destroyComms(state);
}

ncclResult_t cocclCommGetZeroCtaComm(
    ncclComm_t comm, ncclComm_t* zeroCtaComm) {
  pthread_mutex_lock(&commLock);
  cocclCommState& state = comms.at(comm);
  if (state.zeroCtaComm != nullptr) {
    *zeroCtaComm = state.zeroCtaComm;
    pthread_mutex_unlock(&commLock);
    return ncclSuccess;
  }
  pthread_mutex_unlock(&commLock);

  ncclComm_t created = nullptr;
  NCCLCHECK(splitZeroCtaComm(comm, 0, comm->rank, &created));

  pthread_mutex_lock(&commLock);
  comms.at(comm).zeroCtaComm = created;
  pthread_mutex_unlock(&commLock);
  *zeroCtaComm = created;
  return ncclSuccess;
}

ncclResult_t cocclCommGetHierarchicalComms(
    ncclComm_t comm, cocclHierarchicalComms* hierarchy) {
  pthread_mutex_lock(&commLock);
  cocclCommState& state = comms.at(comm);
  if (state.intraComm != nullptr) {
    *hierarchy = {comm, state.intraComm, state.interComm};
    pthread_mutex_unlock(&commLock);
    return ncclSuccess;
  }
  pthread_mutex_unlock(&commLock);

  cocclCommState created;
  const int localRanks = comm->localRanks;
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  ncclResult_t result = splitZeroCtaComm(
      comm, comm->rank / localRanks, comm->rank, &created.intraComm);
  if (result == ncclSuccess) {
    result = splitZeroCtaComm(
        comm, comm->rank % localRanks, comm->rank, &created.interComm);
  }
  if (result != ncclSuccess) {
    (void)destroyComms(created);
    return result;
  }
  pthread_mutex_lock(&commLock);
  cocclCommState& saved = comms.at(comm);
  saved.intraComm = created.intraComm;
  saved.interComm = created.interComm;
  pthread_mutex_unlock(&commLock);
  *hierarchy = {comm, created.intraComm, created.interComm};
  return ncclSuccess;
}
