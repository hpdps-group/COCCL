#ifndef COCCL_SENDRECV_H_
#define COCCL_SENDRECV_H_

#include "coccl_prepared_call.h"

ncclResult_t cocclExecuteSendRecv(const cocclPreparedCall* prepared);
ncclResult_t cocclExecuteSendRecvBatch(
    const cocclPreparedCall* calls, size_t count);

#endif
