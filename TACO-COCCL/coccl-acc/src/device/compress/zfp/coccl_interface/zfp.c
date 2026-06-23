#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "compressor.h"
#include <iostream>
#include "zfp.h"
#include "zfp/internal/zfp/macros.h"
#define __hidden __attribute__ ((visibility("hidden")))

#define ALIGN_SIZE(size, align) \
  size = ((size + (align) - 1) / (align)) * (align);

struct zfpCompConfig{
    int rate=4;
};

__hidden void parseZfpConfig(const char* configFile, void** compConfig, int nodes, int devicesPerNodes){
  *compConfig = (void*) malloc(sizeof(zfpCompConfig));
  zfpCompConfig* config = reinterpret_cast<zfpCompConfig*>(*compConfig);

  config->rate = 4;
  if(!configFile) return;
  // load config from file
  std::pair<const char*, const char*>* configPairs = nullptr;
  int configPairCount = 0;
  loadConfigPair(configFile, &configPairs, &configPairCount);

  if(configPairs == nullptr) return;

  for(int i = 0; i < configPairCount; i++){
    if(strcmp(configPairs[i].first, "rate") == 0){
      char* end;
      int rate = strtol(configPairs[i].second, &end, 10);
      if(*end == '\0'){
        config->rate = static_cast<int>(rate);
      }
    }
  }
}


cudaError_t launchZfpCompress(const void* orgbuff, void** compbuff, const size_t orgChunkCount, ncclDataType_t orgDayatype,
                                  size_t* compChunkCount, ncclDataType_t* compDatatype, const size_t numChunks, void* config, 
                                  cudaMemPool_t compMemPool, cudaStream_t stream)
{
  int rate = 4;
  if(config != NULL){
    zfpCompConfig* zfp_config = (zfpCompConfig*)config;
    rate = zfp_config->rate;
  }
  
  zfp_stream* zfp = zfp_stream_open(NULL);
  zfp_field* field = zfp_field_alloc();
  zfp_field_set_pointer(field, const_cast<void*>(orgbuff));
  zfp_type type = zfp_type_none;
  if(orgDayatype == ncclFloat32)
    type = zfp_type_float;
  else if(orgDayatype == ncclFloat16 || orgDayatype == ncclHalf)
    type = zfp_type_float16;
  else if(orgDayatype == ncclBfloat16)
    type = zfp_type_bfloat16;

  zfp_field_set_type(field, type);
  zfp_field_set_size_1d(field, orgChunkCount * numChunks);

  zfp_stream_set_rate(zfp, rate, type, 1, zfp_false);
  if (!zfp_stream_set_execution(zfp, zfp_exec_cuda)) {
    fprintf(stderr, "cuda execution not available\n");
    return cudaErrorUnknown;
  }
  size_t bufsize = zfp_stream_maximum_size(zfp, field);
  // rank0 128 + 24 
  // rank1 128 + 24 
  // rank2 128 + 24 
  // rank3 128 + 24 

  // allgather 

  // rank 0 
  // 128 + 24 | 128 + 24 | 128 + 24 | 128 + 24
  
  if(*compbuff == NULL) 
    cudaMallocAsync((void**)compbuff, bufsize, stream);  
  
  bitstream* in_stream = stream_open(*compbuff, orgChunkCount * numChunks / (8 / rate));
  zfp_stream_set_bit_stream(zfp, in_stream);
  // size_t aaa = zfp_stream_maximum_size(zfp, field);
  // printf("bufsize %lld\n", bufsize);
  zfp->exec.params = (cudaStream_t *) malloc(sizeof(cudaStream_t*));
  zfp->exec.params = &stream;
  size_t zfpsize = zfp_compress(zfp, field);
  // printf("rate %d orgsize %lld zfpsize %lld\n", rate, orgChunkCount * numChunks * 2, zfpsize);
  if (zfpsize == 0) {
    fprintf(stderr, "compression failed\n");
    return cudaErrorUnknown;
  }
  *compChunkCount = zfpsize / numChunks;
  *compDatatype = ncclUint8;
  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(in_stream);

  return cudaGetLastError();
}


cudaError_t launchZfpDecompress(void* decompbuff, const void* compbuff, const size_t decompChunkCount, ncclDataType_t decompDatatype, 
                                    const size_t compChunkCount, ncclDataType_t compDatatype, const size_t numChunks, void* config, 
                                    cudaStream_t stream)
{
  int rate = 4;
  if(config != NULL){
    zfpCompConfig* zfp_config = (zfpCompConfig*)config;
    rate = zfp_config->rate;
  }
  
  zfp_stream* zfp = zfp_stream_open(NULL);
  zfp_field* field = zfp_field_alloc();
  zfp_type type = zfp_type_none;
  if(decompDatatype == ncclFloat32)
    type = zfp_type_float;
  else if(decompDatatype == ncclFloat16 || decompDatatype == ncclHalf)
    type = zfp_type_float16;
  else if(decompDatatype == ncclBfloat16)
    type = zfp_type_bfloat16;

  bitstream* in_stream = stream_open(const_cast<void*>(compbuff), compChunkCount * numChunks);
  zfp_stream_set_bit_stream(zfp, in_stream);
  zfp_field_set_type(field, type);
  zfp_field_set_size_1d(field, decompChunkCount * numChunks);
  zfp_stream_set_rate(zfp, rate, type, 1, zfp_false);
  if (!zfp_stream_set_execution(zfp, zfp_exec_cuda)) {
    fprintf(stderr, "cuda execution not available\n");
    return cudaErrorUnknown;
  }
  zfp_stream_rewind(zfp);
  zfp_field_set_pointer(field, decompbuff);
  zfp->exec.params = (cudaStream_t *) malloc(sizeof(cudaStream_t *));
  zfp->exec.params = &stream;
  size_t zfpdesize =  zfp_decompress(zfp, field);

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(in_stream);
  
  return cudaGetLastError();
}


extern "C" const ncclCompressor_t zfp_compressor{
  .name = "zfp_compressor",
  .compress = launchZfpCompress,
  .decompress = launchZfpDecompress,
  .decompReduce = nullptr,
  .decompReduceComp = nullptr,
  .parseConfig = parseZfpConfig
};
