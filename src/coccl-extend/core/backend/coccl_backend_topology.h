#ifndef COCCL_BACKEND_TOPOLOGY_H_
#define COCCL_BACKEND_TOPOLOGY_H_

#include "nccl.h"

#include <stddef.h>
#include <stdint.h>

enum class cocclAutotuneTopologyOperation : uint8_t;

double cocclBackendEstimateStage(
    ncclComm_t comm, cocclAutotuneTopologyOperation operation, size_t bytes);

#endif
