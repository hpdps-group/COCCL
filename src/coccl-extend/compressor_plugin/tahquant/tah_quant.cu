#include "dequantization_utils.h"
#include "memory_access_utils.h"
#include "ds_kernel_utils.h"
#include "quantization.h"
#include "quantization_utils.h"
#include "reduction_utils.h"
#include <cuda_runtime.h> 
#include <cuda_fp16.h>  
// #include "device.h"
// #include "checks.h"
// #include "debug.h"
#include "compressor.h"
#include <iostream>

#define __hidden __attribute__ ((visibility("hidden")))


#define ALIGN_SIZE(size, align) \
  size = ((size + (align) - 1) / (align)) * (align);

struct tahQuantConfig{
    /* custom configs*/
    // normal
    int groupCount=2048;
    int quantBits=8;
    bool hadamard = false;
    bool pivotSwap = false;
    quantize::Type quantType = quantize::Type::Symmetric;

} ;
inline int ncclTypeSize(ncclDataType_t type) {
    switch (type) {
    case ncclInt8:
    case ncclUint8:
      return 1;
    case ncclFloat16:
    #if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
    #endif
      return 2;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64:
      return 8;
    default:
      return -1;
    }
  }

__hidden void parseTahQuantConfig(const char* configFile, void** compConfig, int nodes, int devicesPerNodes){
  // alloc memory for config
  *compConfig = (void*) malloc(sizeof(tahQuantConfig));
  tahQuantConfig* config = reinterpret_cast<tahQuantConfig*>(*compConfig);
  // default values
  config->groupCount = 2048;
  config->hadamard = false;
  config->quantBits = 8;
  config->quantType = quantize::Type::Symmetric;
  config->pivotSwap = false;
 
  if(!configFile) return;
  // load config from file
  std::pair<const char*, const char*>* configPairs = nullptr;
  int configPairCount = 0;
  loadConfigPair(configFile, &configPairs, &configPairCount); 
  if(configPairs == nullptr) return;
  // get configs
  for(int i = 0; i < configPairCount; i++){
      // groupCounts
      if(strcmp(configPairs[i].first, "groupCount") == 0){
          char* end;
          long groupCount = strtol(configPairs[i].second, &end, 10);
          if(*end == '\0'){
              config->groupCount = static_cast<int>(groupCount);
          }
      }
      // quantBits
      if(strcmp(configPairs[i].first, "quantBits") == 0){
          char* end;
          int quantBits = strtol(configPairs[i].second, &end, 10);
          if(*end == '\0'){
              config->quantBits = static_cast<int>(quantBits);
          }
      }
      // hadamard
      if(strcmp(configPairs[i].first, "hadamard") == 0){
          config->hadamard = (strcmp(configPairs[i].second, "1") == 0);
      }
      // pivotSwap
      if(strcmp(configPairs[i].first, "pivotSwap") == 0){
        config->pivotSwap = (strcmp(configPairs[i].second, "1") == 0);
      }
      // quantType
      if(strcmp(configPairs[i].first, "quantType") == 0){
          if(strcmp(configPairs[i].second, "Symmetric") == 0)config->quantType = quantize::Type::Symmetric;
          else if(strcmp(configPairs[i].second, "Asymmetric") == 0)config->quantType = quantize::Type::Asymmetric;
      }
  }
 
}


#define GETSTOCHCOMPBUFF()                                                                                  \
    size_t quanScales = 8 / quantBits;                                                                      \
    size_t quantBytes = groupCount * sizeof(int8_t) / quanScales;                                           \
    size_t posBytes = sizeof(int64_t);                                                                      \
    size_t paramsBytes = orgDayatype == ncclDataType_t::ncclFloat32 ?                                       \
       (quantType == quantize::Type::Symmetric ? 1 : 2) * sizeof(float) : 2 * sizeof(float);                \
    if(hadamard && pivotSwap)                                                                               \
        *compChunkCount = groupsPerChunk * (quantBytes + paramsBytes + posBytes + 1);                       \
    else *compChunkCount = groupsPerChunk * (quantBytes + paramsBytes);                                     \
    if(*compbuff == nullptr || *compbuff == NULL)                                                           \
    {                                                                                                       \
        if(compMemPool == nullptr || compMemPool == NULL)                                                   \
            cudaMallocAsync((void**)compbuff, (*compChunkCount) * numChunks, stream);                       \
        else                                                                                                \
            cudaMallocFromPoolAsync((void**)compbuff, (*compChunkCount) * numChunks, compMemPool, stream);  \
    }

cudaError_t launchTahQuant(const void* orgbuff, void** compbuff, const size_t orgChunkCount, ncclDataType_t orgDayatype,
                                  size_t* compChunkCount, ncclDataType_t* compDatatype, const size_t numChunks, const int rank, void* config, 
                                  cudaMemPool_t compMemPool, cudaStream_t stream)
{
    int groupCount = 128;
    int quantBits = 8;
    quantize::Type quantType = quantize::Type::Symmetric;
    bool hadamard = false;
    bool pivotSwap = false;
    if(config != NULL || config != nullptr){
        tahQuantConfig* quanConfig = (tahQuantConfig*)config;
        groupCount =  quanConfig->groupCount;
        quantBits =  quanConfig->quantBits;
        quantType = quanConfig->quantType;
        hadamard = quanConfig->hadamard;
        pivotSwap = quanConfig->pivotSwap;
    }
    // when hadamard groupsize large than 128, there will be some error
    if(hadamard && groupCount > 128) groupCount = 128;

    int groupsPerChunk = (orgChunkCount + groupCount - 1) / groupCount;
    *compDatatype = ncclDataType_t::ncclInt8;
    // printf("orgChunkCount %zu numGroups %d compChunkCount %zu\n", orgChunkCount, numGroups, *compChunkCount);
    GETSTOCHCOMPBUFF();
    void* tempbuff = (char*)(*compbuff) + orgChunkCount * numChunks * ncclTypeSize(orgDayatype);
    float* params = nullptr;
    size_t posOffset = groupsPerChunk * (quantBytes + paramsBytes);
    size_t flagOffset = groupsPerChunk * (quantBytes + paramsBytes + sizeof(int64_t));
    if(pivotSwap)
        cudaMemcpyAsync(tempbuff, orgbuff, orgChunkCount * numChunks * ncclTypeSize(orgDayatype), cudaMemcpyDeviceToDevice, stream);
    if(orgDayatype == ncclDataType_t::ncclFloat32){
        if(!hadamard)
            launch_quant((int8_t*)*compbuff, params, (float*)orgbuff, numChunks * groupsPerChunk, groupCount, quantBits, quantType, stream);
        else{
            if(!pivotSwap)
                launch_quant_ht((int8_t*)*compbuff, params, (float*)orgbuff, numChunks * groupsPerChunk, groupCount, quantBits, quantType, stream);
            else {
                launch_pivot_swap_experimental((float*)tempbuff, numChunks * groupsPerChunk, groupCount, (int64_t*)((char*)*compbuff + posOffset),  (bool*)((char*)*compbuff + flagOffset), stream);
                launch_quant_heuristic_ht((int8_t*)*compbuff, params, (float*)tempbuff, numChunks * groupsPerChunk, groupCount, quantBits, quantType, (bool*)((char*)*compbuff + flagOffset), stream);
            }
        }
    }
    else if(orgDayatype == ncclDataType_t::ncclFloat16){
        if(!hadamard)
            launch_quant((int8_t*)*compbuff, params, (__half*)orgbuff, numChunks * groupsPerChunk, groupCount, quantBits, quantType, stream);
        else{
            if(!pivotSwap)
                launch_quant_ht((int8_t*)*compbuff, params, (__half*)orgbuff, numChunks * groupsPerChunk, groupCount, quantBits, quantType, stream);
            else {
                launch_pivot_swap_experimental((__half*)tempbuff, numChunks * groupsPerChunk, groupCount, (int64_t*)((char*)*compbuff + posOffset), (bool*)((char*)*compbuff + flagOffset), stream);
                launch_quant_heuristic_ht((int8_t*)*compbuff, params, (__half*)tempbuff, numChunks * groupsPerChunk, groupCount, quantBits, quantType, (bool*)((char*)*compbuff + flagOffset), stream);
            }
        }
    }
    else if(orgDayatype == ncclDataType_t::ncclBfloat16){
        if(!hadamard){
            launch_quant((int8_t*)*compbuff, params, (__nv_bfloat16*)orgbuff, numChunks * groupsPerChunk, groupCount, quantBits, quantType, stream);
        }
        else{
            if(!pivotSwap)
                launch_quant_ht((int8_t*)*compbuff, params, (__nv_bfloat16*)orgbuff, numChunks * groupsPerChunk, groupCount, quantBits, quantType, stream);
            else {
                launch_pivot_swap_experimental((__nv_bfloat16*)tempbuff, numChunks * groupsPerChunk, groupCount, (int64_t*)((char*)*compbuff + posOffset), (bool*)((char*)*compbuff + flagOffset), stream);
                launch_quant_heuristic_ht((int8_t*)*compbuff, params, (__nv_bfloat16*)tempbuff, numChunks * groupsPerChunk, groupCount, quantBits, quantType, (bool*)((char*)*compbuff + flagOffset), stream);
            }
        }
    }
    return cudaGetLastError();
}
__hidden cudaError_t launchDequantize(void* decompbuff, const void* compbuff, const size_t decompChunkCount, ncclDataType_t decompDatatype, 
                                    const size_t compChunkCount, ncclDataType_t compDatatype, const size_t numChunks, void* config, 
                                    cudaStream_t stream)
{

    int groupCount = 128;
    int quantBits = 8;
    quantize::Type quantType = quantize::Type::Symmetric;
    bool hadamard = false;
    bool pivotSwap = false;

    if(config != NULL || config != nullptr){
        tahQuantConfig* quanConfig = (tahQuantConfig*)config;
        groupCount = quanConfig->groupCount;
        quantBits = quanConfig->quantBits;
        quantType = quanConfig->quantType;
        hadamard = quanConfig->hadamard;
        pivotSwap = quanConfig->pivotSwap;

    }
    if(hadamard && groupCount > 128) groupCount = 128;

    int groupsPerChunk = (decompChunkCount + groupCount - 1) / groupCount;
    int64_t totalCounts = (int64_t)numChunks * decompChunkCount;
    float* params =nullptr;
    // printf("decompChunkCount %zu numGroups %d totalCounts %ld compChunkCount %zu\n", decompChunkCount, numGroups, totalCounts, compChunkCount);
    size_t quanScales = 8 / quantBits; 
    size_t quantBytes = groupCount * sizeof(int8_t) / quanScales;
    size_t paramsBytes = decompDatatype == ncclDataType_t::ncclFloat32 ?
       (quantType == quantize::Type::Symmetric ? 1 : 2) * sizeof(float) : 2 * sizeof(float);
    size_t posOffset = groupsPerChunk * (quantBytes + paramsBytes);
    size_t flagOffset = groupsPerChunk * (quantBytes + paramsBytes + sizeof(int64_t));

    if(decompDatatype == ncclDataType_t::ncclFloat32){
        if(!hadamard)
                launch_dequantize_kernel((float*)decompbuff, (const int8_t*)compbuff, params, quantType, quantBits, groupCount, totalCounts, stream);
        else {
            if(!pivotSwap)
                launch_dequantize_ht_kernel((float*)decompbuff, (const int8_t*)compbuff, params, quantType, quantBits, groupCount, totalCounts, stream);
            else {
                launch_dequantize_heuristic_ht_kernel((float*)decompbuff, (const int8_t*)compbuff, params, quantType, quantBits, groupCount, totalCounts, (bool*)((char*)compbuff + flagOffset), stream);
                launch_swap_back_experimental((float*)decompbuff, groupsPerChunk * numChunks, groupCount, (int64_t*)((char*)compbuff + posOffset), (bool*)((char*)compbuff + flagOffset), stream);
            }
        }
    }
    else if(decompDatatype == ncclDataType_t::ncclFloat16){
        if(!hadamard)
            launch_dequantize_kernel((__half*)decompbuff, (const int8_t*)compbuff, params, quantType, quantBits, groupCount, totalCounts, stream);
        else{
            if(!pivotSwap)
                launch_dequantize_ht_kernel((__half*)decompbuff, (const int8_t*)compbuff, params, quantType, quantBits, groupCount, totalCounts, stream);
            else {
                launch_dequantize_heuristic_ht_kernel((__half*)decompbuff, (const int8_t*)compbuff, params, quantType, quantBits, groupCount, totalCounts, (bool*)((char*)compbuff + flagOffset), stream);
                launch_swap_back_experimental((__half*)decompbuff, groupsPerChunk * numChunks, groupCount, (int64_t*)((char*)compbuff + posOffset), (bool*)((char*)compbuff + flagOffset), stream);
            }
        }
    }
    else if(decompDatatype == ncclDataType_t::ncclBfloat16){
        if(!hadamard)
            launch_dequantize_kernel((__nv_bfloat16*)decompbuff, (const int8_t*)compbuff, params, quantType, quantBits, groupCount, totalCounts, stream);
        else{
            if(!pivotSwap)
                launch_dequantize_ht_kernel((__nv_bfloat16*)decompbuff, (const int8_t*)compbuff, params, quantType, quantBits, groupCount, totalCounts, stream);
            else {
                launch_dequantize_heuristic_ht_kernel((__nv_bfloat16*)decompbuff, (const int8_t*)compbuff, params, quantType, quantBits, groupCount, totalCounts, (bool*)((char*)compbuff + flagOffset), stream);
                launch_swap_back_experimental((__nv_bfloat16*)decompbuff, groupsPerChunk * numChunks, groupCount, (int64_t*)((char*)compbuff + posOffset), (bool*)((char*)compbuff + flagOffset), stream);
            }
        }
    }
  
    return cudaGetLastError();
}


__hidden cudaError_t launchDequanReduceQuan(const void* compbuff, void** recompbuff, const size_t compChunkCount, ncclDataType_t compDatatype,
                                    size_t* reCompChunkCount, ncclDataType_t* reCompDatatype, const size_t numChunks, void* config,
                                    cudaMemPool_t compMemPool, cudaStream_t stream)
{
    int inQuantBits = 8;
    int outQuantBits = 8;
    int inGroupCount = 2048;
    int outGroupCount = 2048;
    quantize::Type quantType = quantize::Type::Symmetric;
    if(config != NULL || config != nullptr){
        tahQuantConfig* quanConfig = (tahQuantConfig*)config;
        quantType = quanConfig->quantType;
        inQuantBits = quanConfig->quantBits;
        outQuantBits = quanConfig->quantBits;
        inGroupCount = quanConfig->groupCount;
        outGroupCount = quanConfig->groupCount;
    }
    int inGroupBytes = inGroupCount / (8 / inQuantBits); // number of Bytes
    int outGroupBytes = outGroupCount / (8 / outQuantBits); // number of Bytes
    int paramsBytes = (quantType == quantize::Type::Symmetric ? 1 : 2) * sizeof(float); 
    // one group is GroupBytes + paramsBytes
    int inChunkGroups = (compChunkCount + (inGroupBytes + paramsBytes) - 1) / (inGroupBytes + paramsBytes);
    int outChunkGroups = (inChunkGroups * inGroupCount + outGroupCount - 1) / outGroupCount;
    int64_t inChunkBytes = (int64_t)inChunkGroups * inGroupBytes;

    *reCompDatatype = compDatatype;
    *reCompChunkCount = (outGroupBytes + paramsBytes) * outChunkGroups;
    if(*recompbuff == nullptr || *recompbuff == NULL)                                                           
    {                                                                                                       
        if(compMemPool == nullptr || compMemPool == NULL)                                                   
            cudaMallocAsync((void**)recompbuff, (*reCompChunkCount), stream);                       
        else                                                                                                
            cudaMallocFromPoolAsync((void**)recompbuff, (*reCompChunkCount), compMemPool, stream); 
    }
    // printf("compChunkCount %zu inGroupBytes: %d, outGroupBytes: %d, paramsBytes: %d, inChunkGroups: %d, outChunkGroups: %d reCompChunkCount %zu\n",
    //     compChunkCount, inGroupBytes, outGroupBytes, paramsBytes, inChunkGroups, outChunkGroups,*reCompChunkCount);

    float* inputScales =nullptr;
    float* outScales = nullptr;

    launch_dequant_reduce_quant((int8_t*)(*recompbuff), outScales, (const int8_t*)compbuff, inputScales, 
                        numChunks, inQuantBits, outQuantBits, quantType, outChunkGroups, outGroupBytes, 
                        inChunkBytes, inChunkGroups, inGroupBytes, stream);
    return cudaGetLastError();
}


extern "C" const ncclCompressor_t tahquant{
  .name = "tahquant",
  .compress = launchTahQuant,
  .decompress = launchDequantize,
  .decompReduce = nullptr,
  .decompReduceComp = launchDequanReduceQuan,
  .parseConfig = parseTahQuantConfig
};
