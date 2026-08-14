#ifndef NCCL_COMPRESS_H_
#define NCCL_COMPRESS_H_

#include "device.h"
#include "core.h"
#include "argcheck.h"
#include "core/compression/coccl_compressor_runtime.h"

// Direct single-compressor API used by the pipeline after resolving a policy
// once. Byte capacity and typed metadata are both explicit.
ncclResult_t ncclCompress(
    const cocclCompressorHandle& compressor,
    const cocclCompressorView& input,
    cocclCompressorView* output, int rank, cudaStream_t stream);
ncclResult_t ncclDecompress(
    const cocclCompressorHandle& compressor,
    const cocclCompressorView& input,
    cocclCompressorView* output, cudaStream_t stream);
ncclResult_t ncclDecompressReduce(
    const cocclCompressorHandle& compressor, ncclComm_t ownerComm,
    const cocclCompressorView& input,
    cocclCompressorView* output, size_t reduceChunks,
    cudaStream_t stream);
ncclResult_t ncclDecompReduceComp(
    const cocclCompressorHandle& compressor, ncclComm_t ownerComm,
    const cocclCompressorView& input,
    cocclCompressorView* output, size_t reduceChunks,
    ncclDataType_t originalDatatype, size_t originalElements,
    cudaStream_t stream);

#endif
