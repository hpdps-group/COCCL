#ifndef COCCL_ALLGATHER_H_
#define COCCL_ALLGATHER_H_

#include "coccl_prepared_call.h"

ncclResult_t cocclExecuteAllGather(const cocclPreparedCall* prepared);

#endif
