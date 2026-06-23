#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "compressor.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include "ans/GpuANSEncode.h"
#include "ans/GpuANSCodec.h"
#include "ans/GpuANSDecode.h"
using namespace multibyte_ans;
#define __hidden __attribute__ ((visibility("hidden")))

#define ALIGN_SIZE(size, align) \
  size = ((size + (align) - 1) / (align)) * (align);

struct pansCompConfig{
    int precision=10;
};

__hidden void parsePansConfig(const char* configFile, void** compConfig, int nodes, int devicesPerNodes){
  *compConfig = (void*) malloc(sizeof(pansCompConfig));
  pansCompConfig* config = reinterpret_cast<pansCompConfig*>(*compConfig);

  config->precision = 10;
  if(!configFile) return;
  // load config from file
  std::pair<const char*, const char*>* configPairs = nullptr;
  int configPairCount = 0;
  loadConfigPair(configFile, &configPairs, &configPairCount);

  if(configPairs == nullptr) return;

  for(int i = 0; i < configPairCount; i++){
    if(strcmp(configPairs[i].first, "precision") == 0){
      char* end;
      int precision = strtol(configPairs[i].second, &end, 10);
      if(*end == '\0'){
        config->precision = static_cast<int>(precision);
      }
    }
  }
}

__thread uint32_t* outCompressedSize = nullptr;
__thread uint4* table_dev = nullptr;
__thread uint32_t* tempHistogram_dev = nullptr;
__thread uint8_t* compressedBlocks_dev = nullptr;
__thread uint32_t* compressedWords_dev = nullptr;
__thread uint32_t* compressedWordsPrefix_dev = nullptr;
__thread uint8_t* tempPrefixSum_dev = nullptr;

cudaError_t launchPansCompress(const void* orgbuff, void** compbuff, const size_t orgChunkCount, ncclDataType_t orgDayatype,
                                  size_t* compChunkCount, ncclDataType_t* compDatatype, const size_t numChunks, void* config, 
                                  cudaMemPool_t compMemPool, cudaStream_t stream)
{
  int precision = 10;
  if(config != NULL){
    pansCompConfig* pans_config = (pansCompConfig*)config;
    precision = pans_config->precision;
  }

  size_t batchSize = orgChunkCount;
  if(orgDayatype == ncclFloat32)
    batchSize *= 4;
  else if(orgDayatype == ncclFloat16 || orgDayatype == ncclHalf)
    batchSize *= 2;
  else if(orgDayatype == ncclBfloat16)
    batchSize *= 2;
  uint32_t maxUncompressedWords = batchSize / sizeof(ANSDecodedT);
  uint32_t maxNumCompressedBlocks =
    (maxUncompressedWords + kDefaultBlockSize - 1) / kDefaultBlockSize;
  auto sizeRequired =
    getBatchExclusivePrefixSumTempSize(
      maxNumCompressedBlocks);
   uint32_t uncoalescedBlockStride =
      getMaxBlockSizeUnCoalesced(kDefaultBlockSize);
  if(outCompressedSize == nullptr) {
    cudaMalloc(&outCompressedSize, sizeof(uint32_t));
    cudaMalloc(&table_dev, sizeof(uint4) * kNumSymbols);
    cudaMalloc(&tempHistogram_dev, sizeof(uint32_t) * kNumSymbols);
    cudaMalloc(&compressedBlocks_dev, sizeof(uint8_t) * maxNumCompressedBlocks * uncoalescedBlockStride);
    cudaMalloc(&compressedWords_dev, sizeof(uint32_t) * maxNumCompressedBlocks);
    cudaMalloc(&compressedWordsPrefix_dev, sizeof(uint32_t) * maxNumCompressedBlocks);
    cudaMalloc(&tempPrefixSum_dev, sizeof(uint8_t) * sizeRequired);
  }
  
  if(*compbuff == NULL) 
    cudaMallocAsync((void**)compbuff, static_cast<uint64_t>(getMaxCompressedSize(batchSize)), stream);
  
  *compDatatype = ncclUint8;
  *compChunkCount = batchSize;
  uint8_t* inPtrs = (uint8_t*) orgbuff;
  uint8_t* encPtrs = (uint8_t*) *compbuff;
  for(size_t i =0 ;i < numChunks; i++){
    ansEncode(
          maxUncompressedWords,
          maxNumCompressedBlocks,
          table_dev,
          tempHistogram_dev,
          uncoalescedBlockStride,
          compressedBlocks_dev,
          compressedWords_dev,
          compressedWordsPrefix_dev,
          sizeRequired,
          tempPrefixSum_dev,
          precision,
          inPtrs,//已经在dev
          batchSize,
          encPtrs,//GPU
          outCompressedSize,//GPU
          stream);
    uint32_t outsize;
    cudaMemcpy(&outsize, outCompressedSize, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    inPtrs += batchSize;
    encPtrs += outsize;
    *compChunkCount = outsize > *compChunkCount ? *compChunkCount : outsize;
  }
  
  return cudaGetLastError();
}


cudaError_t launchPansDecompress(void* decompbuff, const void* compbuff, const size_t decompChunkCount, ncclDataType_t decompDatatype, 
                                    const size_t compChunkCount, ncclDataType_t compDatatype, const size_t numChunks, void* config, 
                                    cudaStream_t stream)
{
  int precision = 10;
  if(config != NULL){
    pansCompConfig* pans_config = (pansCompConfig*)config;
    precision = pans_config->precision;
  }
  size_t batchSize = decompChunkCount;
  if(decompDatatype == ncclFloat32)
    batchSize *= 4;
  else if(decompDatatype == ncclFloat16 || decompDatatype == ncclHalf)
    batchSize *= 2;
  else if(decompDatatype == ncclBfloat16)
    batchSize *= 2;
  uint8_t* encPtrs = (uint8_t*) compbuff;
  uint8_t* decPtrs = (uint8_t*) decompbuff;

  for(size_t i =0 ; i<numChunks; i++){
    ansDecode(
      precision,
      encPtrs, 
      decPtrs,
      stream);
    encPtrs += compChunkCount;
    decPtrs += batchSize;

  }

  return cudaGetLastError();
}


extern "C" const ncclCompressor_t pans{
  .name = "pans",
  .compress = launchPansCompress,
  .decompress = launchPansDecompress,
  .decompReduce = nullptr,
  .decompReduceComp = nullptr,
  .parseConfig = parsePansConfig
};
