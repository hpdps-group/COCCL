#ifndef COCCL_PRIMITIVES_INTERNAL_H_
#define COCCL_PRIMITIVES_INTERNAL_H_

#include <stddef.h>
#include "nccl_comp_wrapper.h"
#include "nccl.h"
#include "argcheck.h"
#include "enqueue.h"
#include "compress.h"
#include "reduce_extend.h"
#include "compressor.h"
#include "coccl_buffer_management.h"
#include "coccl_comm.h"
#include "coccl_pipeline.h"
#include "../../graph/topo.h"

// Common implementation dependencies for primitives expressed through the
// current COCCL pipeline. Legacy-only state lives in
// coccl_old_impl_internal.h and is not exposed to new primitive code.

#endif
