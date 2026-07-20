#ifndef COCCL_AUTOTUNE_H_
#define COCCL_AUTOTUNE_H_

#include "coccl_comm_op.h"
#include "nccl.h"

struct cocclRuntimeArgs;

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

enum cocclAlgorithmKind {
  cocclAlgorithmNone = 0,
  cocclAlgorithmReduceScatterOneShot,
  cocclAlgorithmReduceScatterTwoShot,
  cocclAlgorithmAllReduceOneShot,
  cocclAlgorithmAllReduceTwoShot,
  cocclAlgorithmAllReduceTripleShot,
};

struct cocclAlgorithmDecision {
  cocclAlgorithmKind algorithm = cocclAlgorithmNone;
  double oneShotUs = 0.0;
  double twoShotUs = 0.0;
  double tripleShotUs = 0.0;
  bool usedModel = false;
};

struct cocclAutotuneProfileOptions {
  bool profileReduceScatter = false;
  bool profileAllReduce = false;
};

// The first eligible communicator cooperatively fills each process-wide model.
// A later communicator only fills model categories that could not previously be
// sampled, such as inter-node performance after a single-node initialization.
ncclResult_t cocclAutotuneProfile(
    ncclComm_t comm, const cocclAutotuneProfileOptions* options);

// Pure CPU hot-path selector. It never launches CUDA or NCCL work.
ncclResult_t cocclSelectAlgorithm(const cocclRuntimeArgs* args,
                                  cocclAlgorithmDecision* decision);

#endif
