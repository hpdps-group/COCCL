#include "compressor.h"

#define __hidden __attribute__ ((visibility("hidden")))

__hidden ncclResult_t exampleCompress(const void* orgbuff, void** compbuff, const size_t orgChunkCount, ncclDataType_t orgDayatype, 
    const size_t* compChunkCount, ncclDataType_t* compDatatype, const size_t numChunks, ncclCompressConfig config, cudaStream_t stream)
{
    return ncclSuccess;
}

__hidden ncclResult_t exampleDecompress(void *decompbuff, const void* compbuff, const size_t decompChunkCount, ncclDataType_t decompDatatype,
    const size_t compChunkCount, ncclDataType_t compDatatype, const size_t numChunks, ncclCompressConfig config, cudaStream_t stream)
{
    return ncclSuccess;
}

const ncclCompressor_t exampleCompressor{
    .name = "example",
    .compress = exampleCompress,
    .decompress = exampleDecompress
};