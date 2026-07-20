#ifndef COCCL_COMM_H_
#define COCCL_COMM_H_

#include "coccl_comm_op.h"
#include "coccl_training_assist.h"
#include "compressor.h"
#include "nccl.h"
#include "nccl_common.h"

struct cocclRuntimeArgs;

// Opaque per-communicator state builder APIs. coccl_comm.cc owns the hidden
// cocclComm layout; runtime only supplies config and compressor chain entries.
ncclResult_t cocclCommCreate(ncclComm_t comm);
// Heavy resources are detached under the registry lock and destroyed after the
// lock is released, because split-comm teardown can re-enter NCCL/COCCL paths.
struct cocclCommDetachedResources;
ncclResult_t cocclCommDestroy(ncclComm_t comm, bool* registryEmpty,
                              cocclCommDetachedResources** detachedResources);
ncclResult_t cocclCommDestroyDetachedResources(cocclCommDetachedResources* detachedResources);
// Publishes a fully initialized sidecar. Before commit, only the default chain
// may be used by init-time profiling.
ncclResult_t cocclCommCommit(ncclComm_t comm);

// The default chain is reserved for pre-commit autotune profiling. Committed
// communication requires an explicit normal op or training role/op chain.
// These append APIs take ownership of non-null config pointers with free().
ncclResult_t cocclCommAppendDefaultCompressor(ncclComm_t comm, ncclCompressor_t* compressor, void* config);
ncclResult_t cocclCommResetOpChain(
    ncclComm_t comm, ncclCommOp_t op, size_t thresholdBytes);
ncclResult_t cocclCommAppendOpCompressor(ncclComm_t comm, ncclCommOp_t op, ncclCompressor_t* compressor, void* config);
ncclResult_t cocclCommCopyOpChain(ncclComm_t comm, ncclCommOp_t dstOp, ncclCommOp_t srcOp);

// Training-role chains remain private to cocclComm. Runtime builds one chain
// per role/op so a role chooses the compressor while the op chooses its config.
ncclResult_t cocclCommResetTrainingRoleChain(
    ncclComm_t comm, cocclTrainingRole role, ncclCommOp_t op,
    size_t thresholdBytes);
ncclResult_t cocclCommAppendTrainingRoleCompressor(
    ncclComm_t comm, cocclTrainingRole role, ncclCommOp_t op,
    ncclCompressor_t* compressor, void* config);

bool cocclCommAvailable(ncclComm_t comm);

// Unified routing predicate. totalBytes is the complete logical input size,
// not the per-rank count accepted by AllGather/ReduceScatter. Reduction
// collectives require ncclSum; non-reduction operations ignore reductionOp.
bool cocclCommShouldCompress(
    ncclComm_t comm, ncclCommOp_t compressorOp, cocclTrainingRole role,
    size_t totalBytes, ncclDataType_t datatype, ncclRedOp_t reductionOp);

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

// Visitor-style compressor chain access keeps chain storage private to
// coccl_comm.cc while letting compress.cc execute plugin callbacks.
typedef ncclResult_t (*cocclCompressorVisitor)(ncclCompressor_t* compressor, void* config, void* context);
ncclResult_t cocclVisitCompressorChain(ncclComm_t comm, ncclCommOp_t op, bool reverse,
                                       cocclCompressorVisitor visitor, void* context);

#endif
