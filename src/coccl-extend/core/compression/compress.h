#ifndef NCCL_COMPRESS_H_
#define NCCL_COMPRESS_H_

#include "device.h"
#include "core.h"
#include "argcheck.h"
#include "core/compression/coccl_compressor_runtime.h"

ncclResult_t ncclCompress(
    void* compressor, const cocclCompressorView& input,
    cocclCompressorView* output, int rank, cudaStream_t stream);
ncclResult_t ncclDecompress(
    void* compressor, const cocclCompressorView& input,
    cocclCompressorView* output, cudaStream_t stream);
ncclResult_t ncclDecompressReduce(
    void* compressor, ncclComm_t ownerComm,
    const cocclCompressorView& input, cocclCompressorView* output,
    size_t reduceChunks, cudaStream_t stream);
ncclResult_t ncclDecompReduceComp(
    void* decoder, void* encoder, ncclComm_t ownerComm,
    const cocclCompressorView& input, cocclCompressorView* output,
    size_t reduceChunks, ncclDataType_t originalDatatype,
    size_t originalElements, cudaStream_t stream);

#endif
