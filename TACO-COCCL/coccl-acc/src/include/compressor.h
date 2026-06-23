#ifndef NCCL_COMPRESSOR_H_
#define NCCL_COMPRESSOR_H_

#include "nccl.h"
#include "compress_utils.h"
#include "align.h"
// API to be implemented by external compressor
typedef struct {
    //Name of the compressor
    const char* name;
    // Compress 
    // Inputs:
    //   - orgbuff: original buffer
    //   - orgChunkCount: original buffer chunk count (chunkCount = bufferCount / comm->nRanks)
    //   - orgDatatype: original datatype (currently the compressor supports fp32, fp16 and bf16)
    //   - numChunks: the number of chunks to compress (usually it is comm->nRanks)
    //   - config: Configuration parameters of the compressor (it is char* and the compressor need parse it)
    //   - compMemPool: memory pool for compressed buffer
    //   - stream: cuda stream
    //
    // Outputs:
    //   - compbuff: pointer of compressed buffer (it could be allocated by the compressor)
    //   - compChunkCount: pointer of compressed buffer chunk count
    //   - compDatatype: pointer of compressed datatype (compressor will return the datatype of the compressed buffer)
    //
    cudaError_t (*compress)(const void* orgbuff, void** compbuff, const size_t orgChunkCount, ncclDataType_t orgDayatype, 
                size_t* compChunkCount, ncclDataType_t* compDatatype, const size_t numChunks, void* config, 
                cudaMemPool_t compMemPool, cudaStream_t stream);

    // Decompress 
    // Inputs:
    //   - compbuff: compressed buffer
    //   - decompChunkCount: decompressed buffer chunk count (usually it is equal to orgChunkCount)
    //   - decompDatatype: decompressed datatype (currently the compressor supports fp32, fp16 and bf16)
    //   - compChunkCount: compressed buffer chunk count 
    //   - compDatatype: compressed datatype
    //   - numChunks: the number of chunks to compress (usually it is comm->nRanks)
    //   - config: Configuration parameters of the compressor (it is char* and the compressor need parse it)
    //   - stream: cuda stream
    //
    // Outputs:
    //   - decompbuff: decompressed buffer
    //
    cudaError_t (*decompress)(void* decompbuff, const void* compbuff, const size_t decompChunkCount, ncclDataType_t decompDatatype,
                const size_t compChunkCount, ncclDataType_t compDatatype, const size_t numChunks, void* config,
                cudaStream_t stream);

    cudaError_t (*decompReduce)(void* reducebuff, const void* compbuff, const size_t compChunkCount, ncclDataType_t compDatatype,
                const size_t reduceChunkCount, ncclDataType_t reduceDataType, const size_t numChunks, void* config,
                cudaStream_t stream);

    cudaError_t (*decompReduceComp)(const void* compbuff, void** recompbuff, const size_t compChunkCount, ncclDataType_t compDatatype,
                size_t* reCompChunkCount, ncclDataType_t* reCompDatatype, const size_t numChunks, void* config,
                cudaMemPool_t compMemPool, cudaStream_t stream);

    /* users need to implement this function to parse the config !*/
    // Inputs:
    //   - configFile: the yaml format config file path
    // Outputs:
    //   - compConfig: the parsed config
    void (*parseConfig)(const char* configFile, void** compConfig, int nodes, int devicesPerNodes);
    
} ncclCompressor;

typedef ncclCompressor ncclCompressor_t;



#endif
