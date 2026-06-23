#include "dequantization_utils.h"
#include "memory_access_utils.h"
#include "ds_kernel_utils.h"
#include "quantization.h"
#include "quantization_utils.h"
#include "reduction_utils.h"
#include <cuda_runtime.h> 
#include <cuda_fp16.h>  
#include "nccl.h"
// #include "device.h"
// #include "checks.h"
// #include "debug.h"
#include "compressor.h"

#define __hidden __attribute__ ((visibility("hidden")))

#define ALIGN_SIZE(size, align) \
  size = ((size + (align) - 1) / (align)) * (align);

struct sdp4bitConfig{
    /* custom configs*/
    // normal
    int groupCount=2048;
    int quantBits=8;
    bool hadamard = false;
    quantize::Type quantType = quantize::Type::Symmetric;
    // gradient config
    int inQuantBits = 0;
    int outQuantBits = 0;
    int inGroupCount = 0;
    int outGroupCount = 0;
    // intra and inter
    bool intraAndInter = false;
    // swizzle
    int nodes = 1;
    int devicesPerNodes = 4;
    int pipelineSize = 1;
} ;

__hidden void parseSDP4BitConfig(const char* configFile, void** compConfig, int nodes, int devicesPerNodes){
    // alloc memory for config
    *compConfig = (void*) malloc(sizeof(sdp4bitConfig));
    sdp4bitConfig* config = reinterpret_cast<sdp4bitConfig*>(*compConfig);
    // default values
    config->groupCount = 2048;
    config->hadamard = false;
    config->quantBits = 8;
    config->quantType = quantize::Type::Symmetric;
    config->inQuantBits = 0;
    config->outQuantBits = 0;
    config->inGroupCount = 0;
    config->outGroupCount = 0;
    config->intraAndInter = false;
    config->pipelineSize = 1;
    config->nodes = nodes;
    config->devicesPerNodes = devicesPerNodes;
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
        // quantType
        if(strcmp(configPairs[i].first, "quantType") == 0){
            if(strcmp(configPairs[i].second, "Symmetric") == 0)config->quantType = quantize::Type::Symmetric;
            else if(strcmp(configPairs[i].second, "Asymmetric") == 0)config->quantType = quantize::Type::Asymmetric;
        }
        // inQuanBits
        if(strcmp(configPairs[i].first, "inQuanBits") == 0){
            char* end;
            int inQuantBits = strtol(configPairs[i].second, &end, 10);
            if(*end == '\0'){
                config->inQuantBits = static_cast<int>(inQuantBits);
            }
        }
        // outQuanBits
        if(strcmp(configPairs[i].first, "outQuanBits") == 0){
            char* end;
            int outQuantBits = strtol(configPairs[i].second, &end, 10);
            if(*end == '\0'){
                config->outQuantBits = static_cast<int>(outQuantBits);
            }
        }
        // inGroupCount
        if(strcmp(configPairs[i].first, "inGroupCount") == 0){
            char* end;
            int inGroupCount = strtol(configPairs[i].second, &end, 10);
            if(*end == '\0'){
                config->inGroupCount = static_cast<int>(inGroupCount);
            }
        }   
        // outGroupCount
        if(strcmp(configPairs[i].first, "outGroupCount") == 0){
            char* end;
            int outGroupCount = strtol(configPairs[i].second, &end, 10);
            if(*end == '\0'){
                config->outGroupCount = static_cast<int>(outGroupCount);
            }
        }
        // intraAndInter
        if(strcmp(configPairs[i].first, "intraAndInter") == 0){
            config->intraAndInter = (strcmp(configPairs[i].second, "1") == 0);
        }
        // pipelineSize
        if(strcmp(configPairs[i].first, "pipelineSize") == 0){
            char* end;
            int pipelineSize = strtol(configPairs[i].second, &end, 10);
            if(*end == '\0'){
                config->pipelineSize = static_cast<int>(pipelineSize);
            }
        }
    }
   
}


#define GETSTOCHCOMPBUFF()                                                                                  \
    size_t quanScales = 8 / quantBits;                                                                      \
    size_t quantBytes = groupCount * sizeof(int8_t) / quanScales;                                           \
    size_t paramsBytes = orgDayatype == ncclDataType_t::ncclFloat32 ?                                       \
       (quantType == quantize::Type::Symmetric ? 1 : 2) * sizeof(float) : 2 * sizeof(float);                \
    *compChunkCount = numGroups * (quantBytes + paramsBytes);                                               \
    if(*compbuff == nullptr || *compbuff == NULL)                                                           \
    {                                                                                                       \
        if(compMemPool == nullptr || compMemPool == NULL)                                                   \
            cudaMallocAsync((void**)compbuff, (*compChunkCount) * numChunks, stream);                       \
        else                                                                                                \
            cudaMallocFromPoolAsync((void**)compbuff, (*compChunkCount) * numChunks, compMemPool, stream);  \
    }

cudaError_t launchSwizzleQuan(const void* orgbuff, void** compbuff, const size_t orgChunkCount, ncclDataType_t orgDayatype,
                                    size_t* compChunkCount, ncclDataType_t* compDatatype, const size_t numChunks, void* config, 
                                    cudaMemPool_t compMemPool, cudaStream_t stream)
{
    int groupCount = 2048;
    int quantBits = 8;
    quantize::Type quantType = quantize::Type::Symmetric;
    bool hadamard = false;
    int nodes = 1;
    int devicesPerNodes = 4;
    int pipelineSize = 1;
    if(config != NULL || config != nullptr){
        sdp4bitConfig* quanConfig = (sdp4bitConfig*)config;
        groupCount =  (quanConfig->inGroupCount == 0) ? quanConfig->groupCount: quanConfig->inGroupCount;
        quantBits = (quanConfig->inQuantBits == 0) ? quanConfig->quantBits: quanConfig->inQuantBits;
        quantType = quanConfig->quantType;
        hadamard = quanConfig->hadamard;
        nodes = quanConfig->nodes;
        devicesPerNodes = quanConfig->devicesPerNodes;
        pipelineSize = quanConfig->pipelineSize;
    }
    // when hadamard groupsize large than 128, there will be some error
    if(hadamard && groupCount > 128) groupCount = 128;

    int numGroups = (orgChunkCount + groupCount - 1) / groupCount;
    
    *compDatatype = ncclDataType_t::ncclInt8;
    GETSTOCHCOMPBUFF();
    float* params =nullptr;
    // swizzle
    if(orgDayatype == ncclDataType_t::ncclFloat32){
        if(!hadamard)
            launch_swizzled_quant((int8_t*)*compbuff, params, (float*)orgbuff, quantBits, quantType, numChunks * numGroups, groupCount, 
                        pipelineSize, nodes, devicesPerNodes, stream);
        else
            launch_swizzled_quant_ht((int8_t*)*compbuff, params, (float*)orgbuff, quantBits, quantType, numChunks * numGroups, groupCount, 
                        pipelineSize, nodes, devicesPerNodes, stream);
    }
    else if(orgDayatype == ncclDataType_t::ncclFloat16){
        if(!hadamard)
            launch_swizzled_quant((int8_t*)*compbuff, params, (__half*)orgbuff, quantBits, quantType, numChunks * numGroups, groupCount, 
                        pipelineSize, nodes, devicesPerNodes, stream);
        else
            launch_swizzled_quant_ht((int8_t*)*compbuff, params, (__half*)orgbuff, quantBits, quantType, numChunks * numGroups, groupCount, 
                        pipelineSize, nodes, devicesPerNodes, stream);
    }
    
    return cudaGetLastError();
}

cudaError_t launchStochasticQuan(const void* orgbuff, void** compbuff, const size_t orgChunkCount, ncclDataType_t orgDayatype,
                                    size_t* compChunkCount, ncclDataType_t* compDatatype, const size_t numChunks, void* config, 
                                    cudaMemPool_t compMemPool, cudaStream_t stream)
{
    int groupCount = 2048;
    int quantBits = 8;
    quantize::Type quantType = quantize::Type::Symmetric;
    bool hadamard = false;
    if(config != NULL || config != nullptr){
        sdp4bitConfig* quanConfig = (sdp4bitConfig*)config;
        groupCount = (quanConfig->inGroupCount == 0) ? quanConfig->groupCount: quanConfig->inGroupCount;
        quantBits = (quanConfig->inQuantBits == 0) ? quanConfig->quantBits: quanConfig->inQuantBits;
        quantType = quanConfig->quantType;
        hadamard = quanConfig->hadamard;
    }
    // when hadamard groupsize large than 128, there will be some error
    if(hadamard && groupCount > 128) groupCount = 128;

    int numGroups = (orgChunkCount + groupCount - 1) / groupCount;
    *compDatatype = ncclDataType_t::ncclInt8;
    // printf("orgChunkCount %zu numGroups %d compChunkCount %zu\n", orgChunkCount, numGroups, *compChunkCount);
    GETSTOCHCOMPBUFF();
    float* params =nullptr;
    if(orgDayatype == ncclDataType_t::ncclFloat32){
        if(!hadamard)
            launch_quant((int8_t*)*compbuff, params, (float*)orgbuff, numChunks * numGroups, groupCount, quantBits, quantType, stream);
        else
            launch_quant_ht((int8_t*)*compbuff, params, (float*)orgbuff, numChunks * numGroups, groupCount, quantBits, quantType, stream);

    }
    else if(orgDayatype == ncclDataType_t::ncclFloat16){
        if(!hadamard)
            launch_quant((int8_t*)*compbuff, params, (__half*)orgbuff, numChunks * numGroups, groupCount, quantBits, quantType, stream);
        else
            launch_quant_ht((int8_t*)*compbuff, params, (__half*)orgbuff, numChunks * numGroups, groupCount, quantBits, quantType, stream);
    }
    else if(orgDayatype == ncclDataType_t::ncclBfloat16){
        if(!hadamard)
            launch_quant((int8_t*)*compbuff, params, (__nv_bfloat16*)orgbuff, numChunks * numGroups, groupCount, quantBits, quantType, stream);
        else
            launch_quant_ht((int8_t*)*compbuff, params, (__nv_bfloat16*)orgbuff, numChunks * numGroups, groupCount, quantBits, quantType, stream);
    }
    return cudaGetLastError();
}

__hidden cudaError_t launchQuantize(const void* orgbuff, void** compbuff, const size_t orgChunkCount, ncclDataType_t orgDayatype,
                                    size_t* compChunkCount, ncclDataType_t* compDatatype, const size_t numChunks, void* config, 
                                    cudaMemPool_t compMemPool, cudaStream_t stream)
{
    bool intraAndInter = false;
    if(config != NULL || config != nullptr){
        sdp4bitConfig* quanConfig = (sdp4bitConfig*)config;
        intraAndInter = quanConfig->intraAndInter;
    }
    if(intraAndInter)
        launchSwizzleQuan(orgbuff, compbuff, orgChunkCount, orgDayatype, compChunkCount, compDatatype, numChunks, config,
                        compMemPool, stream);
    else 
        launchStochasticQuan(orgbuff, compbuff, orgChunkCount, orgDayatype, compChunkCount, compDatatype, numChunks, config,
                        compMemPool, stream);
    return cudaGetLastError();
}

__hidden cudaError_t launchDequantize(void* decompbuff, const void* compbuff, const size_t decompChunkCount, ncclDataType_t decompDatatype, 
                                    const size_t compChunkCount, ncclDataType_t compDatatype, const size_t numChunks, void* config, 
                                    cudaStream_t stream)
{

    int groupCount = 2048;
    int quantBits = 8;
    quantize::Type quantType = quantize::Type::Symmetric;
    if(config != NULL || config != nullptr){
        sdp4bitConfig* quanConfig = (sdp4bitConfig*)config;
        groupCount =  (quanConfig->outGroupCount == 0) ? quanConfig->groupCount : quanConfig->outGroupCount;
        quantBits = (quanConfig->outQuantBits == 0)? quanConfig->quantBits : quanConfig->outQuantBits;
        quantType = quanConfig->quantType;
    }
    int numGroups = (decompChunkCount + groupCount - 1) / groupCount;
    int64_t totalCounts = (int64_t)numChunks * decompChunkCount;
    float* params =nullptr;
    // printf("decompChunkCount %zu numGroups %d totalCounts %ld compChunkCount %zu\n", decompChunkCount, numGroups, totalCounts, compChunkCount);

    if(decompDatatype == ncclDataType_t::ncclFloat32)
        launch_dequantize_kernel((float*)decompbuff, (const int8_t*)compbuff, params, quantType, quantBits, groupCount, totalCounts, stream);
    else if(decompDatatype == ncclDataType_t::ncclFloat16)
        launch_dequantize_kernel((__half*)decompbuff, (const int8_t*)compbuff, params, quantType, quantBits, groupCount, totalCounts, stream);
    else if(decompDatatype == ncclDataType_t::ncclBfloat16)
        launch_dequantize_kernel((__nv_bfloat16*)decompbuff, (const int8_t*)compbuff, params, quantType, quantBits, groupCount, totalCounts, stream);
  
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
        sdp4bitConfig* quanConfig = (sdp4bitConfig*)config;
        quantType = quanConfig->quantType;
        inQuantBits = (quanConfig->inQuantBits == 0) ? quanConfig->quantBits: quanConfig->inQuantBits;
        outQuantBits = (quanConfig->outQuantBits == 0) ? quanConfig->quantBits: quanConfig->outQuantBits;
        inGroupCount = (quanConfig->inGroupCount == 0) ? quanConfig->groupCount: quanConfig->inGroupCount;
        outGroupCount = (quanConfig->outGroupCount == 0) ? quanConfig->groupCount: quanConfig->outGroupCount;
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


__hidden cudaError_t launchDequanReduce(void* reducebuff, const void* compbuff, const size_t compChunkCount, ncclDataType_t compDatatype,
                                    const size_t reduceChunkCount, ncclDataType_t reduceDataType, const size_t numChunks, void* config,
                                    cudaStream_t stream)
{
    int quantBits = 8;
    int groupCount = 2048;
    quantize::Type quantType = quantize::Type::Symmetric;
    bool hadamard = false;
    if(config != NULL || config != nullptr){
        sdp4bitConfig* quanConfig = (sdp4bitConfig*)config;
        quantType = quanConfig->quantType;
        quantBits = (quanConfig->outQuantBits == 0)? quanConfig->quantBits : quanConfig->outQuantBits;
        groupCount = (quanConfig->outGroupCount == 0) ? quanConfig->groupCount : quanConfig->outGroupCount;
        hadamard = quanConfig->hadamard;
    }
    // when hadamard groupsize large than 128, there will be some error
    if(hadamard && groupCount > 128) groupCount = 128;
    // printf("dequanReduce quantBits %d\n", quantBits);


    const float* input_scales = nullptr;
    int numGroups = (reduceChunkCount + groupCount - 1) / groupCount;
    int groupBytes = groupCount / (8 / quantBits);
    int64_t chunkBytes = (int64_t)numGroups * groupBytes;
    if(!hadamard)
        launch_dequant_reduce((float*)reducebuff, (const int8_t*)compbuff, input_scales, numChunks, quantBits, quantType,
                    numGroups, groupCount * sizeof(float), chunkBytes, numGroups, groupBytes, stream);
    else
        launch_dequant_reduce_ht((float*)reducebuff, (const int8_t*)compbuff, input_scales, numChunks, quantBits, quantType,
                    numGroups, groupCount * sizeof(float), chunkBytes, numGroups, groupBytes, stream);
    return cudaGetLastError();
}


extern "C" const ncclCompressor_t sdp4bit{
    .name = "sdp4bit",
    .compress = launchQuantize,
    .decompress = launchDequantize,
    .decompReduce = launchDequanReduce,
    .decompReduceComp = launchDequanReduceQuan,
    .parseConfig = parseSDP4BitConfig
};



// debug

 // float *params_h = (float*)malloc(paramsBytes);
        // // memset(params_h,0,sizeof(float)* 2 * groups);
        // uint8_t *quan = (uint8_t*)malloc(quanBytes);
        // float *host_input = (float*)malloc(orgChunkCount* sizeof(float));

 // cudaMemcpyAsync(host_input, inputbuff, orgChunkCount* sizeof(float), cudaMemcpyDeviceToHost, stream);
            // // cudaMemcpyAsync(params_h, paramsbuff, paramsBytes, cudaMemcpyDeviceToHost, stream);
            // cudaMemcpyAsync(params_h, paramsbuff, paramsBytes, cudaMemcpyDeviceToHost, stream);
            // cudaMemcpyAsync(quan, quanbuff, quanBytes, cudaMemcpyDeviceToHost, stream);
            // // cudaMemcpyAsync(output, dequant_data, inputCount* sizeof(float), cudaMemcpyDeviceToHost, stream);

            // for(int r=0;r<numGroups;r++){            
            //   for(int i=0;i<groupCounts;i++){
            //     printf("ori: %f, quan: %u", host_input[r*groupCounts+i], quan[r*groupCounts+i]);
            //   }
            //     printf("groupid %d: \n", r);
            // }
            // for(int r=0;r<2 * numGroups;r++){
            //       printf("params_h: %f\n",params_h[r]);
            // }
            



   // float *params_h = (float*)malloc(paramsBytes);
        // // memset(params_h,0,sizeof(float)* 2 * groups);
        // uint8_t *quan = (uint8_t*)malloc(quanBytes);
        // float *dequan = (float*)malloc(decompChunkCount* sizeof(float));


// cudaMemcpyAsync(dequan, outbuff, decompChunkCount* sizeof(float), cudaMemcpyDeviceToHost, stream);
            // // cudaMemcpyAsync(params_h, paramsbuff, paramsBytes, cudaMemcpyDeviceToHost, stream);
            // cudaMemcpyAsync(params_h, paramsbuff, paramsBytes, cudaMemcpyDeviceToHost, stream);
            // cudaMemcpyAsync(quan, quanbuff, quanBytes, cudaMemcpyDeviceToHost, stream);
            // // cudaMemcpyAsync(output, dequant_data, inputCount* sizeof(float), cudaMemcpyDeviceToHost, stream);
            // cudaStreamSynchronize(stream);

            // for(int r=0;r<numGroups;r++){            
            //   for(int i=0;i<groupCounts;i++){
            //     printf("quan: %u, ori: %f\n",  quan[r*groupCounts+i], dequan[r*groupCounts+i]);
            //   }
            //     printf("groupid %d: \n", r);
            // }
            // for(int r=0;r<2 * numGroups;r++){
            //       printf("params_h: %f\n",params_h[r]);
            // }