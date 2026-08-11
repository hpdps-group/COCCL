#ifndef NCCL_COMPRESS_H_
#define NCCL_COMPRESS_H_

#include "device.h"
#include "core.h"
#include "argcheck.h"
#include "runtime/coccl_operation.h"
#include "runtime/coccl_compressor_runtime.h"

// Direct single-compressor API used by the pipeline after resolving a policy
// once. Byte capacity and typed metadata are both explicit.
ncclResult_t ncclCompress(
    const cocclCompressorHandle& compressor,
    const cocclCompressorDataView& input,
    cocclCompressorOutputView* output, int rank, cudaStream_t stream);
ncclResult_t ncclDecompress(
    const cocclCompressorHandle& compressor,
    const cocclCompressorDataView& input,
    cocclCompressorOutputView* output, cudaStream_t stream);
ncclResult_t ncclDecompressReduce(
    const cocclCompressorHandle& compressor, ncclComm_t ownerComm,
    const cocclCompressorDataView& input,
    cocclCompressorOutputView* output, size_t reduceChunks,
    cudaStream_t stream);
ncclResult_t ncclDecompReduceComp(
    const cocclCompressorHandle& compressor, ncclComm_t ownerComm,
    const cocclCompressorDataView& input,
    cocclCompressorOutputView* output, size_t reduceChunks,
    ncclDataType_t originalDatatype, size_t originalElements,
    cudaStream_t stream);

// Compatibility API for legacy primitives. Each entry resolves exactly one
// configured policy; new primitives receive a prepared compressor explicitly.
ncclResult_t ncclCompress(const void* orgbuff, void** compbuff, const size_t orgChunkCount, ncclDataType_t orgDayatype,
    size_t* compChunkCount, ncclDataType_t* compDatatype, const size_t numChunks, const int rank, ncclComm_t comm,
    cocclPolicyKey policy, cudaStream_t stream);

ncclResult_t ncclDecompress(void* decompbuff, const void* compbuff, const size_t decompChunkCount, ncclDataType_t decompDatatype,
    const size_t compChunkCount, ncclDataType_t compDatatype, const size_t numChunks, ncclComm_t comm, cocclPolicyKey policy,
    cudaStream_t stream);

ncclResult_t ncclDecompressReduce(void* reducebuff, const void* compbuff, const size_t compChunkCount, ncclDataType_t compDatatype, 
    const size_t reduceChunkCount, ncclDataType_t reduceDataType,  const size_t numChunks, ncclComm_t comm, cocclPolicyKey policy,
    cudaStream_t stream);

ncclResult_t ncclDecompReduceComp(const void* compbuff, void** recompbuff, const size_t orgChunkCount, ncclDataType_t orgDayatype,
    const size_t compChunkCount, ncclDataType_t compDatatype, size_t* reCompChunkCount, ncclDataType_t* reCompDatatype, const size_t numChunks, 
    ncclComm_t comm, cocclPolicyKey policy, cudaStream_t stream);

#endif
