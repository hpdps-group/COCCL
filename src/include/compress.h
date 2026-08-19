#ifndef NCCL_COMPRESS_H_
#define NCCL_COMPRESS_H_

#include "device.h"
#include "core.h"
#include "argcheck.h"
#include "coccl_operation.h"
#include "compressor_plugin/detail/coccl_compressor_abi.h"

enum ncclCommOp{AlltoAll = 0, AlltoAll_Inter = 1, AllReduce = 2, AllReduce_Inter = 3, AllGather = 4, AllGather_Inter = 5, ReduceScatter = 6, ReduceScatter_Inter = 7, SendRecv = 8, SendRecv_BWD = 9};

typedef ncclCommOp ncclCommOp_t;

struct cocclResolvedCompressorPolicy {
  void* compressor;
  size_t thresholdBytes;
};

bool cocclCompressionEnabled();
ncclResult_t cocclResolveCompressorPolicy(
    cocclPolicyKey key, cocclResolvedCompressorPolicy* resolved);
ncclResult_t cocclQueryCompressorEncodedSizeBound(
    const cocclCompressorPlugin* plugin, const void* config,
    cocclCompressorOperation operation, size_t elements, size_t chunks,
    ncclDataType_t datatype, size_t* encodedBytes);
ncclResult_t cocclGetCompressorEncodedSizeBound(
    cocclPolicyKey key, cocclCompressorOperation operation,
    size_t elements, size_t chunks, ncclDataType_t datatype,
    size_t* encodedBytes);
ncclResult_t cocclGetCompressorEncodedSizeBound(
    void* compressor, cocclCompressorOperation operation,
    size_t elements, size_t chunks, ncclDataType_t datatype,
    size_t* encodedBytes);
bool cocclCompressorPolicySupports(
    cocclPolicyKey key, cocclCompressorCapability capability);
bool cocclCompressorSupports(
    void* compressor, cocclCompressorCapability capability);
const cocclCompressorPlugin* cocclCompressorDescriptor(void* compressor);

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


ncclResult_t ncclCompress(const void* orgbuff, void** compbuff, const size_t orgChunkCount, ncclDataType_t orgDayatype,
    size_t* compChunkCount, ncclDataType_t* compDatatype, const size_t numChunks, const int rank, ncclCommOp_t commOp, cudaStream_t stream,
    size_t outputCapacityBytes = 0);

ncclResult_t ncclDecompress(void* decompbuff, const void* compbuff, const size_t decompChunkCount, ncclDataType_t decompDatatype,
    const size_t compChunkCount, ncclDataType_t compDatatype, const size_t numChunks, ncclCommOp_t commOp, cudaStream_t stream);

ncclResult_t ncclDecompressReduce(void* reducebuff, const void* compbuff, const size_t compChunkCount, ncclDataType_t compDatatype, 
    const size_t outputElements, ncclDataType_t reduceDataType,
    const size_t inputChunks, const size_t reduceChunks,
    ncclCommOp_t commOp, cudaStream_t stream, ncclComm_t ownerComm);

ncclResult_t ncclDecompReduceComp(const void* compbuff, void** recompbuff, const size_t originalElements, ncclDataType_t orgDayatype,
    const size_t compChunkCount, ncclDataType_t compDatatype, size_t* reCompChunkCount, ncclDataType_t* reCompDatatype, const size_t numChunks, 
    const size_t reduceChunks, ncclCommOp_t commOp, cudaStream_t stream,
    ncclComm_t ownerComm, size_t outputCapacityBytes = 0);

ncclResult_t ncclCompressInit(const ncclComm_t comm);
ncclResult_t ncclCompressDestroy(const ncclComm_t comm);


#endif
