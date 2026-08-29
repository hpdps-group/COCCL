#include "debug.h"

int ncclDebugLevel __attribute__((weak)) = NCCL_LOG_NONE;
uint64_t ncclDebugMask __attribute__((weak)) = 0;

void __attribute__((weak)) ncclDebugLogInternal(
    ncclDebugLogLevel, unsigned long, const char*, const char*, int,
    const char*, ...) {}
