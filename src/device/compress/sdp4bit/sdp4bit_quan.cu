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
#include <pthread.h>
#include <chrono>
#include <map>


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
    bool subAdd = 0;
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
    config->subAdd = 0;
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
        if(strcmp(configPairs[i].first, "subAdd") == 0){
            config->subAdd = (strcmp(configPairs[i].second, "1") == 0);
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
            launch_swizzled_quant((int8_t*)*compbuff, params, (float*)orgbuff, quantBits, quantType, numChunks, orgChunkCount,
                        groupCount, pipelineSize, nodes, devicesPerNodes, stream);
        else
            launch_swizzled_quant_ht((int8_t*)*compbuff, params, (float*)orgbuff, quantBits, quantType, numChunks, orgChunkCount,
                        groupCount, pipelineSize, nodes, devicesPerNodes, stream);
    } else if(orgDayatype == ncclDataType_t::ncclFloat16){
        if(!hadamard)
            launch_swizzled_quant((int8_t*)*compbuff, params, (__half*)orgbuff, quantBits, quantType, numChunks, orgChunkCount,
                        groupCount, pipelineSize, nodes, devicesPerNodes, stream);
        else
            launch_swizzled_quant_ht((int8_t*)*compbuff, params, (__half*)orgbuff, quantBits, quantType, numChunks, orgChunkCount,
                        groupCount, pipelineSize, nodes, devicesPerNodes, stream);
    } else if(orgDayatype == ncclDataType_t::ncclBfloat16){
        if(!hadamard)
            launch_swizzled_quant((int8_t*)*compbuff, params, (__nv_bfloat16*)orgbuff, quantBits, quantType, numChunks, orgChunkCount,
                        groupCount, pipelineSize, nodes, devicesPerNodes, stream);
        else
            launch_swizzled_quant_ht((int8_t*)*compbuff, params, (__nv_bfloat16*)orgbuff, quantBits, quantType, numChunks, orgChunkCount,
                        groupCount, pipelineSize, nodes, devicesPerNodes, stream);
    }
    
    return cudaGetLastError();
}

cudaError_t launchStochasticQuan(const void* orgbuff, void** compbuff, const size_t orgChunkCount, ncclDataType_t orgDayatype,
                                    size_t* compChunkCount, ncclDataType_t* compDatatype, const size_t numChunks, const int rank, void* config, 
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
            launch_quant((int8_t*)*compbuff, params, (float*)orgbuff, numChunks, orgChunkCount, groupCount, quantBits, quantType, stream);
        else
            launch_quant_ht((int8_t*)*compbuff, params, (float*)orgbuff, numChunks, orgChunkCount, groupCount, quantBits, quantType, stream);
    }
    else if(orgDayatype == ncclDataType_t::ncclFloat16){
        if(!hadamard)
            launch_quant((int8_t*)*compbuff, params, (__half*)orgbuff, numChunks, orgChunkCount, groupCount, quantBits, quantType, stream);
        else
            launch_quant_ht((int8_t*)*compbuff, params, (__half*)orgbuff, numChunks, orgChunkCount, groupCount, quantBits, quantType, stream);
    }
    else if(orgDayatype == ncclDataType_t::ncclBfloat16){
        if(!hadamard)
            launch_quant((int8_t*)*compbuff, params, (__nv_bfloat16*)orgbuff, numChunks, orgChunkCount, groupCount, quantBits, quantType, stream);
        else
            launch_quant_ht((int8_t*)*compbuff, params, (__nv_bfloat16*)orgbuff, numChunks, orgChunkCount, groupCount, quantBits, quantType, stream);
    }
   
    return cudaGetLastError();
}


thread_local std::map<void*, void*> shard_param_lists;
__thread void* temp_list = nullptr;
__thread int temp_rank = 0;
__thread cudaStream_t cpyStream;
__thread cudaEvent_t cpyEvent;
__thread cudaEvent_t commEvent;

void** h_offload_buff = nullptr;
__thread size_t offChunkCount = 0;
__thread int nRanks = 0;
__thread size_t iter =0;
__thread pthread_t tid;


// typedef struct {
//     int cuda_dev;
//     void* h_offload_buff;
//     void* d_offload_buff;
//     size_t total_size;
//     cudaStream_t mainStream;
//     cudaStream_t cpyStream;
// } offloadArgs;

// __thread offloadArgs t_args;

// void* offload_func(void *args){
//     std::this_thread::sleep_for(std::chrono::milliseconds(20));
//     offloadArgs* t_args = (offloadArgs*)args;
//     cudaSetDevice(t_args->cuda_dev);
//     cudaStreamSynchronize(t_args->mainStream);
//     // sleep(2);
//     cudaMemcpyAsync(t_args->h_offload_buff, t_args->d_offload_buff, 
//                 t_args->total_size, cudaMemcpyDeviceToHost, t_args->cpyStream);
//     cudaStreamSynchronize(t_args->cpyStream);
//     return NULL;
// }

cudaError_t launchSubQuan(const void* orgbuff, void** compbuff, const size_t orgChunkCount, ncclDataType_t orgDayatype,
                                    size_t* compChunkCount, ncclDataType_t* compDatatype, const size_t numChunks, const int rank, void* config, 
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
    GETSTOCHCOMPBUFF();
    float* params =nullptr;

    if(orgDayatype == ncclDataType_t::ncclBfloat16){
        if(!hadamard){
            void* recvbuff = (char*)orgbuff - rank * orgChunkCount * 2;
            if(shard_param_lists.find(recvbuff) == shard_param_lists.end()){
                // cudaStreamCreateWithFlags(&cpyStream, cudaStreamNonBlocking);
                // cudaEventCreateWithFlags(&cpyEvent, cudaEventDefault);
                // cudaEventCreateWithFlags(&commEvent, cudaEventDefault);

                cudaMemcpyAsync(*compbuff, orgbuff, orgChunkCount * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
                *compChunkCount = orgChunkCount;
                *compDatatype = orgDayatype;
                temp_rank = rank;
            } else {
                
                // cudaStreamWaitEvent(stream, cpyEvent, 0);
                
                // cudaMallocAsync((void**)&shard_param_list, orgChunkCount * sizeof(__nv_bfloat16), stream);

                // cudaEventRecord(cpyEvent, cpyStream);

                // if(offChunkCount == 0){
                // cudaMemcpyAsync(shard_param_list, (__nv_bfloat16*)h_offload_buff[rank / 4] + rank * orgChunkCount, orgChunkCount * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, stream);
                // cudaMemcpyAsync(shard_param_list, (__nv_bfloat16*)h_offload_buff[rank / 4] + rank * orgChunkCount, orgChunkCount * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, stream);

                
                // }
                // else 
                // {
                //     cudaMemcpyAsync((int8_t*)shard_param_list + orgChunkCount * sizeof(__nv_bfloat16), h_offload_buff, 
                //                             offChunkCount, cudaMemcpyHostToDevice, stream);
                    
                //     // if(rank == 0)
                //     //     printf("offChunkCount %lld orgChunkCount %lld\n", offChunkCount, orgChunkCount);

                //     launch_dequantize_kernel((__nv_bfloat16*)shard_param_list, (int8_t*)shard_param_list + orgChunkCount * sizeof(__nv_bfloat16), 
                //                             params, quantType, 8, 32, orgChunkCount, orgChunkCount, stream);
                // }

                // launch_dequantize_kernel((__nv_bfloat16*)decompbuff, (const int8_t*)compbuff, params, quantType, quantBits, groupCount, decompChunkCount, totalCounts, stream);

                launch_fused_sub_quant_cuda((int8_t*)*compbuff, (__nv_bfloat16*)shard_param_lists[recvbuff] + rank * orgChunkCount, (__nv_bfloat16*)orgbuff, quantBits, 
                                            quantType, groupCount, orgChunkCount, 1, 0, stream);

                // cudaEventRecord(commEvent, stream);
                // cudaStreamWaitEvent(cpyStream, commEvent, 0);

                

            }
        }
        else{
            launch_quant_ht((int8_t*)*compbuff, params, (__half*)orgbuff, numChunks, orgChunkCount, groupCount, quantBits, quantType, stream);
        }
    }
    return cudaGetLastError();
}

__hidden cudaError_t launchQuantize(const void* orgbuff, void** compbuff, const size_t orgChunkCount, ncclDataType_t orgDayatype,
                                    size_t* compChunkCount, ncclDataType_t* compDatatype, const size_t numChunks, const int rank, void* config, 
                                    cudaMemPool_t compMemPool, cudaStream_t stream)
{
    bool intraAndInter = false;
    bool subAdd=false;
    if(config != NULL || config != nullptr){
        sdp4bitConfig* quanConfig = (sdp4bitConfig*)config;
        intraAndInter = quanConfig->intraAndInter;
        subAdd = quanConfig->subAdd;
    }
    if(subAdd)
        launchSubQuan(orgbuff, compbuff, orgChunkCount, orgDayatype, compChunkCount, compDatatype, numChunks, rank, config,
            compMemPool, stream);
    else if(intraAndInter)
        launchSwizzleQuan(orgbuff, compbuff, orgChunkCount, orgDayatype, compChunkCount, compDatatype, numChunks, config,
                        compMemPool, stream);
    else 
        launchStochasticQuan(orgbuff, compbuff, orgChunkCount, orgDayatype, compChunkCount, compDatatype, numChunks, rank, config,
                        compMemPool, stream);
    return cudaGetLastError();
}

pthread_mutex_t pinnedMemLock = PTHREAD_MUTEX_INITIALIZER;


__hidden cudaError_t launchAddDequan(void* decompbuff, const void* compbuff, const size_t decompChunkCount, ncclDataType_t decompDatatype, 
                                    const size_t compChunkCount, ncclDataType_t compDatatype, const size_t numChunks, void* config, 
                                    cudaStream_t stream)
{
    int groupCount = 2048;
    int quantBits = 8;
    // bool subAdd=false;
    quantize::Type quantType = quantize::Type::Symmetric;
    if(config != NULL || config != nullptr){
        sdp4bitConfig* quanConfig = (sdp4bitConfig*)config;
        groupCount =  (quanConfig->outGroupCount == 0) ? quanConfig->groupCount : quanConfig->outGroupCount;
        quantBits = (quanConfig->outQuantBits == 0)? quanConfig->quantBits : quanConfig->outQuantBits;
        quantType = quanConfig->quantType;
    }
    
    if(decompDatatype == ncclDataType_t::ncclBfloat16){
        
        if(shard_param_lists.find(decompbuff) == shard_param_lists.end()){
            // pthread_mutex_lock(&pinnedMemLock);
            // if(h_offload_buff == nullptr)
            //     h_offload_buff = (void**) malloc(numChunks / 4 * sizeof(void*));
            // pthread_mutex_unlock(&pinnedMemLock);
            iter++;

            // if(temp_rank % 4 == 0){
            //     cudaHostAlloc((void **)&h_offload_buff[temp_rank / 4], decompChunkCount * numChunks * sizeof(__nv_bfloat16), cudaHostAllocPortable|cudaHostAllocWriteCombined);
            // }

            // temp_list = decompbuff;
            cudaMallocAsync((void**)&shard_param_lists[decompbuff], decompChunkCount * numChunks * sizeof(__nv_bfloat16), stream);
            cudaMemcpyAsync(decompbuff, compbuff, decompChunkCount * numChunks * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);

            // cudaMemcpyAsync(shard_param_list, (__nv_bfloat16*)compbuff + temp_rank * decompChunkCount, decompChunkCount * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(shard_param_lists[decompbuff], (__nv_bfloat16*)compbuff, decompChunkCount * numChunks * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);

            // cudaMemcpyAsync(h_offload_buff,  (__nv_bfloat16*)compbuff + temp_rank * decompChunkCount, 
            //                     decompChunkCount * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, cpyStream);
            //  cudaMemcpyAsync(h_offload_buff, compbuff, numChunks * decompChunkCount * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, cpyStream);
            // cudaMemcpyAsync((__nv_bfloat16*)h_offload_buff[temp_rank/4] + temp_rank * decompChunkCount, compbuff, numChunks * decompChunkCount * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, cpyStream);

            //  nRanks = numChunks;
            //  cudaEventRecord(cpyEvent, cpyStream);
        } else {
            // if(temp_rank == 0)
            // printf("temp_list_%p_decompbuff_%p_size%lld\n", temp_list, decompbuff,  decompChunkCount * numChunks * sizeof(__nv_bfloat16));
            // cudaMemcpyAsync(decompbuff, param_list, decompChunkCount * numChunks * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
            // cudaMemcpyAsync((__nv_bfloat16*)temp_list + rank * orgChunkCount, shard_param_list, 
            //                             orgChunkCount * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, cpyStream);

            // cudaFreeAsync(shard_param_list, cpyStream);
            // if(temp_rank > 0)
            //     cudaMemcpyAsync((__nv_bfloat16*)decompbuff, h_offload_buff, 
            //                                         temp_rank * decompChunkCount * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, cpyStream);
            // if(temp_rank < nRanks)
            //     cudaMemcpyAsync((__nv_bfloat16*)decompbuff + (temp_rank + 1) * decompChunkCount, (__nv_bfloat16*)h_offload_buff + (temp_rank + 1) * decompChunkCount, 
            //                                         (nRanks - temp_rank - 1) * decompChunkCount * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, cpyStream);

            // cudaEventRecord(cpyEvent, cpyStream);

            // cudaStreamWaitEvent(stream, cpyEvent, 0);

            // launch_fused_dequant_add_cuda((__nv_bfloat16*)decompbuff, (__nv_bfloat16*)decompbuff, (const int8_t*)compbuff, quantBits, quantType, 
            //                 groupCount, decompChunkCount, numChunks, stream);
            launch_fused_dequant_add_cuda((__nv_bfloat16*)decompbuff, (__nv_bfloat16*)shard_param_lists[decompbuff], (const int8_t*)compbuff, quantBits, quantType, 
                            groupCount, decompChunkCount, numChunks, stream);
                            
            cudaMemcpyAsync(shard_param_lists[decompbuff], decompbuff, decompChunkCount * numChunks * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);

            // cudaEventRecord(commEvent, stream);  
            // cudaStreamWaitEvent(cpyStream, commEvent, 0);

            // cudaMemcpyAsync(shard_param_list, (__nv_bfloat16*)decompbuff + temp_rank * decompChunkCount, 
            //                     decompChunkCount * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, cpyStream);

            // launch((__nv_bfloat16*)decompbuff, (__nv_bfloat16*)decompbuff, (const int8_t*)compbuff, quantBits, quantType, 
            //                 groupCount, decompChunkCount, numChunks, stream);

            // size_t paramsBytes = decompDatatype == ncclDataType_t::ncclFloat32 ?                                       
            //                     (quantType == quantize::Type::Symmetric ? 1 : 2) * sizeof(float) : 2 * sizeof(float);   
            // size_t numOffGroups = (decompChunkCount + 32 - 1) / 32;

            // offChunkCount = numOffGroups * (32 + paramsBytes);

            // launch_quant((int8_t*)shard_param_list, nullptr, (__nv_bfloat16*)decompbuff + temp_rank * decompChunkCount, 
            //                     1, decompChunkCount, 32, 8, quantType, cpyStream);

            // cudaMemcpyAsync(h_offload_buff, (__nv_bfloat16*)decompbuff + temp_rank * decompChunkCount, 
            //                     decompChunkCount * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, cpyStream);

            // cudaMemcpyAsync(h_offload_buff, decompbuff, 
            //                     numChunks * decompChunkCount * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, cpyStream);

            // cudaMemcpyAsync(h_offload_buff, (int8_t*)shard_param_list, 
            //                     offChunkCount * sizeof(int8_t), cudaMemcpyDeviceToHost, cpyStream);
            // typedef struct {
            //     int cuda_dev;
            //     void* h_offload_buff;
            //     void* d_offload_buff;
            //     size_t total_size;
            //     cudaStream_t stream;
            // } offloadArgs;

            // cudaSetDevice(t_args.cuda_dev);
            // t_args.cuda_dev = temp_rank % 4;
            // // printf("rank_%d_dev_%d\n", temp_rank,t_args.cuda_dev);
            // t_args.h_offload_buff = h_offload_buff;
            // t_args.d_offload_buff = decompbuff;
            // t_args.total_size =  numChunks * decompChunkCount * sizeof(__nv_bfloat16);
            // t_args.mainStream = stream;
            // t_args.cpyStream = cpyStream;
            // pthread_create(&tid, nullptr, offload_func, &t_args);
            // cudaFreeAsync(shard_param_list, cpyStream);

            // cudaEventRecord(cpyEvent, cpyStream);

        }
    }
    // cudaDeviceSynchronize();

    return cudaGetLastError(); 
}

__hidden cudaError_t launchDequantize(void* decompbuff, const void* compbuff, const size_t decompChunkCount, ncclDataType_t decompDatatype, 
                                    const size_t compChunkCount, ncclDataType_t compDatatype, const size_t numChunks, void* config, 
                                    cudaStream_t stream)
{

    int groupCount = 2048;
    int quantBits = 8;
    bool subAdd=false;
    quantize::Type quantType = quantize::Type::Symmetric;
    bool hadamard = false;

    if(config != NULL || config != nullptr){
        sdp4bitConfig* quanConfig = (sdp4bitConfig*)config;
        groupCount =  (quanConfig->outGroupCount == 0) ? quanConfig->groupCount : quanConfig->outGroupCount;
        quantBits = (quanConfig->outQuantBits == 0)? quanConfig->quantBits : quanConfig->outQuantBits;
        quantType = quanConfig->quantType;
        subAdd = quanConfig->subAdd;
        hadamard = quanConfig->hadamard;
    }

    // int numGroups = (decompChunkCount + groupCount - 1) / groupCount;
    int64_t totalCounts = (int64_t)numChunks * decompChunkCount;
    float* params =nullptr;

    if(subAdd)
        launchAddDequan(decompbuff, compbuff, decompChunkCount, decompDatatype, compChunkCount, compDatatype, numChunks, config, stream);
    else if(decompDatatype == ncclDataType_t::ncclFloat32)
        if(!hadamard)
            launch_dequantize_kernel((float*)decompbuff, (const int8_t*)compbuff, params, quantType, quantBits, groupCount, decompChunkCount, totalCounts, stream);
        else 
            launch_dequantize_ht_kernel((float*)decompbuff, (const int8_t*)compbuff, params, quantType, quantBits, groupCount, decompChunkCount, totalCounts, stream);
    else if(decompDatatype == ncclDataType_t::ncclFloat16)
        if(!hadamard)
            launch_dequantize_kernel((__half*)decompbuff, (const int8_t*)compbuff, params, quantType, quantBits, groupCount, decompChunkCount, totalCounts, stream);
        else
            launch_dequantize_ht_kernel((__half*)decompbuff, (const int8_t*)compbuff, params, quantType, quantBits, groupCount, decompChunkCount, totalCounts, stream);  
    else if(decompDatatype == ncclDataType_t::ncclBfloat16)
        if(!hadamard)
            launch_dequantize_kernel((__nv_bfloat16*)decompbuff, (const int8_t*)compbuff, params, quantType, quantBits, groupCount, decompChunkCount, totalCounts, stream);
        else
            launch_dequantize_ht_kernel((__nv_bfloat16*)decompbuff, (const int8_t*)compbuff, params, quantType, quantBits, groupCount, decompChunkCount, totalCounts, stream);

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

    float* inputScales =nullptr;
    float* outScales = nullptr;

    launch_dequant_reduce_quant((int8_t*)(*recompbuff), outScales, (const int8_t*)compbuff, inputScales, 
                        numChunks, inQuantBits, outQuantBits, quantType, outChunkGroups, outGroupBytes, 
                        inChunkGroups, inGroupBytes, inChunkBytes, stream);

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

    const float* input_scales = nullptr;
    int numGroups = (reduceChunkCount + groupCount - 1) / groupCount;
    int groupBytes = groupCount / (8 / quantBits);
    int64_t chunkBytes = (int64_t)numGroups * groupBytes;
    if(!hadamard)
        launch_dequant_reduce((float*)reducebuff, (const int8_t*)compbuff, input_scales, numChunks, quantBits, quantType,
                    chunkBytes, groupBytes, stream);
    else
        launch_dequant_reduce_ht((float*)reducebuff, (const int8_t*)compbuff, input_scales, numChunks, quantBits, quantType,
                    chunkBytes, groupBytes, stream);
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
