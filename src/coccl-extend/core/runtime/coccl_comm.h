#ifndef COCCL_COMM_H_
#define COCCL_COMM_H_

#include "core/compression/coccl_compressor_runtime.h"
#include "runtime/coccl_operation.h"
#include "core/training/coccl_training_assist.h"
#include "nccl.h"
#include "nccl_common.h"

struct cocclInfo;

// Opaque per-communicator state builder APIs. coccl_comm.cc owns the hidden
// cocclComm layout; runtime only supplies fully constructed compressor handles.
ncclResult_t cocclCommCreate(ncclComm_t comm);
// Heavy resources are detached under the registry lock and destroyed after the
// lock is released, because split-comm teardown can re-enter NCCL/COCCL paths.
struct cocclCommDetachedResources;
ncclResult_t cocclCommDestroy(ncclComm_t comm, bool* registryEmpty,
                              cocclCommDetachedResources** detachedResources);
ncclResult_t cocclCommDestroyDetachedResources(cocclCommDetachedResources* detachedResources);
// Publishes a fully initialized sidecar. Compressor contexts are not visible to
// execution or autotune before the sidecar is committed.
ncclResult_t cocclCommCommit(ncclComm_t comm);
bool cocclCommCommitted(ncclComm_t comm);

ncclResult_t cocclCommSetCompressorPolicy(
    ncclComm_t comm, cocclTrainingRole role, cocclPolicyKey key,
    size_t thresholdBytes, const cocclCompressorHandle& compressor);
ncclResult_t cocclCommCopyCompressorPolicy(
    ncclComm_t comm, cocclTrainingRole role, cocclPolicyKey destination,
    cocclPolicyKey source);

struct cocclResolvedCompressorPolicy {
  cocclCompressorHandle compressor;
  size_t thresholdBytes = 0;
};

// Resolves immutable policy state under cocclCommLock once. Callers perform
// operation/datatype checks outside the registry and can retain the copied
// handle for the complete primitive execution.
ncclResult_t cocclCommResolveCompressorPolicy(
    ncclComm_t comm, cocclTrainingRole role, cocclPolicyKey key,
    cocclResolvedCompressorPolicy* resolved);

struct cocclHierarchicalComms {
  // ownerComm provides device/lifetime ownership for shared COCCL workspace.
  ncclComm_t ownerComm = nullptr;
  // Same-node ranks for the local phase of hierarchical primitives.
  ncclComm_t intraComm = nullptr;
  // Same local-rank-across-nodes ranks for the inter-node phase.
  ncclComm_t interComm = nullptr;
};

// Lazily creates and returns the node-local and same-local-rank
// subcommunicators used by hierarchical COCCL primitives.
ncclResult_t cocclCommGetHierarchicalComms(ncclComm_t comm, cocclHierarchicalComms* resource);

// Resolves the authoritative normal or training role/policy key once. The
// copied handle can be used without holding cocclCommLock. Explicit coccl*Comp*
// primitives use this direct lookup and do not apply the routing threshold.
ncclResult_t cocclCommGetCompressor(
    ncclComm_t comm, cocclPolicyKey key, cocclCompressorHandle* compressor);

#endif
