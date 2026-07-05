#ifndef NCCL_COMPRESS_H_
#define NCCL_COMPRESS_H_

#include "device.h"
#include "core.h"
#include "argcheck.h"
#include "compressor.h"

enum ncclCommOp{AlltoAll = 0, AlltoAll_Inter = 1, AllReduce = 2, AllReduce_Inter = 3, AllGather = 4, AllGather_Inter = 5, ReduceScatter = 6, ReduceScatter_Inter = 7, SendRecv = 8, SendRecv_BWD = 9};

typedef ncclCommOp ncclCommOp_t;


ncclResult_t ncclCompress(const void* orgbuff, void** compbuff, const size_t orgChunkCount, ncclDataType_t orgDayatype,
    size_t* compChunkCount, ncclDataType_t* compDatatype, const size_t numChunks, const int rank, ncclCommOp_t commOp, cudaStream_t stream);

ncclResult_t ncclDecompress(void* decompbuff, const void* compbuff, const size_t decompChunkCount, ncclDataType_t decompDatatype,
    const size_t compChunkCount, ncclDataType_t compDatatype, const size_t numChunks, ncclCommOp_t commOp, cudaStream_t stream);

ncclResult_t ncclDecompressReduce(void* reducebuff, const void* compbuff, const size_t compChunkCount, ncclDataType_t compDatatype, 
    const size_t reduceChunkCount, ncclDataType_t reduceDataType,  const size_t numChunks, ncclCommOp_t commOp, cudaStream_t stream);

ncclResult_t ncclDecompReduceComp(const void* compbuff, void** recompbuff, const size_t orgChunkCount, ncclDataType_t orgDayatype,
    const size_t compChunkCount, ncclDataType_t compDatatype, size_t* reCompChunkCount, ncclDataType_t* reCompDatatype, const size_t numChunks, 
    ncclCommOp_t commOp, cudaStream_t stream);

ncclResult_t ncclCompressInit(const ncclComm_t comm);


#endif