#ifndef NCCL_COMPRESS_H_
#define NCCL_COMPRESS_H_

#include "device.h"
#include "core.h"
#include "argcheck.h"
#include "coccl_comm_op.h"
#include "compressor.h"

// Private execution API used by COCCL primitives after runtime dispatch has
// selected a compressor chain. Public NCCL ABI remains in nccl.h.in.
// Compress/decompress dispatchers run the compressor list associated with
// commOp. Compression walks the list forward; decompression walks it backward.
ncclResult_t ncclCompress(const void* orgbuff, void** compbuff, const size_t orgChunkCount, ncclDataType_t orgDayatype,
    size_t* compChunkCount, ncclDataType_t* compDatatype, const size_t numChunks, const int rank, ncclComm_t comm,
    ncclCommOp_t commOp, cudaStream_t stream);

ncclResult_t ncclDecompress(void* decompbuff, const void* compbuff, const size_t decompChunkCount, ncclDataType_t decompDatatype,
    const size_t compChunkCount, ncclDataType_t compDatatype, const size_t numChunks, ncclComm_t comm, ncclCommOp_t commOp,
    cudaStream_t stream);

ncclResult_t ncclDecompressReduce(void* reducebuff, const void* compbuff, const size_t compChunkCount, ncclDataType_t compDatatype, 
    const size_t reduceChunkCount, ncclDataType_t reduceDataType,  const size_t numChunks, ncclComm_t comm, ncclCommOp_t commOp,
    cudaStream_t stream);

ncclResult_t ncclDecompReduceComp(const void* compbuff, void** recompbuff, const size_t orgChunkCount, ncclDataType_t orgDayatype,
    const size_t compChunkCount, ncclDataType_t compDatatype, size_t* reCompChunkCount, ncclDataType_t* reCompDatatype, const size_t numChunks, 
    ncclComm_t comm, ncclCommOp_t commOp, cudaStream_t stream);

#endif
