#ifndef COCCL_AUTOTUNE_H_
#define COCCL_AUTOTUNE_H_

#include "compression/coccl_compressor.h"
#include "nccl.h"

struct cocclPreparedCall;

// Closed-form latency model fitted during the first eligible communicator
// initialization. Times are expressed in microseconds and bytes are the input
// to the fitted operation.
struct cocclLinearModel {
  double alphaUs = 0.0;
  double betaUsPerByte = 0.0;
  bool valid = false;
};

struct cocclCodecModel {
  cocclLinearModel time;
  double compressionRatio = 1.0;
  bool valid = false;
};

// Registers a process-lifetime plugin from the parsed catalog for independent
// benchmarking. Registration does not create an executable compressor policy.
ncclResult_t cocclAutotuneRegisterEnabledCompressor(
    const cocclCompressorPlugin* compressor);

// Lazily fills missing process-wide models. measurementComm only supplies a
// CUDA device and, when needed, ranks for P2P sampling; no model belongs to it.
// Once compressor/intra models are attempted, later communicators are O(1)
// no-ops except the first multi-node communicator may add the inter-node model.
ncclResult_t cocclAutotuneEnsureGlobalModels(ncclComm_t measurementComm);

// Pure CPU hot-path selector. It never launches CUDA or NCCL work.
ncclResult_t cocclSelectAlgorithm(cocclPreparedCall* prepared);

#endif
