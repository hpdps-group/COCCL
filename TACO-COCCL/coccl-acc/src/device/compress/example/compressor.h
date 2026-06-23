#ifndef NCCL_COMPRESSOR_H_
#define NCCL_COMPRESSOR_H_

#include "nccl.h"

// chunkcounts must be align to 32 bytes
#define CHUNK_ALIGN 32

typedef enum {COMPRESSOR_A=1, COMPRESSOR_B=2} compressorType;

typedef struct {
    int quanBits=8;
    int groupCount=2048;
    cudaMemPool_t compMemPool=NULL;
    /* custom configs*/
} ncclCompressConfig;


// API to be implemented by external compressor
typedef struct {
    //Name of the compressor
    const char* name;

    
    //compress
    ncclResult_t (*compress)(const void* orgbuff, void** compbuff, const size_t orgChunkCount, ncclDataType_t orgDayatype, 
        const size_t* compChunkCount, ncclDataType_t* compDatatype, const size_t numChunks, ncclCompressConfig config, cudaStream_t stream);
    
    //decompress
    ncclResult_t (*decompress)(void *decompbuff, const void* compbuff, const size_t decompChunkCount, ncclDataType_t decompDatatype,
        const size_t compChunkCount, ncclDataType_t compDatatype, const size_t numChunks, ncclCompressConfig config, cudaStream_t stream);
    
} ncclCompressor;

typedef ncclCompressor ncclCompressor_t;
typedef ncclCompressConfig ncclCompressConfig_t;




#endif
