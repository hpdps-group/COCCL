#include "coccl_alloc.h"
#include <map>

#define COMPBUFF_EXCESS_SIZE_T 16

// __thread void* cbuff = nullptr;

typedef struct{
    void* handle = nullptr;
    size_t maxBytes = 0;
} cbuffMeta;
thread_local std::map<void**, cbuffMeta> handleMap;

// __thread void* cbuffHandle = nullptr;
__thread size_t curBytes = 0;

ncclResult_t allocAndRegBuff(void** buff, size_t expectBytes, ncclComm_t comm){
    void *tempbuff = nullptr;
    cbuffMeta* meta = &handleMap[buff];
    if(*buff == nullptr || expectBytes > meta->maxBytes){
      CUDACHECK(cudaDeviceSynchronize());
      if(meta->handle != nullptr && expectBytes > meta->maxBytes){
          CUDACHECK(cudaMalloc(&tempbuff, meta->maxBytes));
          CUDACHECK(cudaMemcpy(tempbuff, *buff, meta->maxBytes, cudaMemcpyDeviceToDevice));
          if(comm != nullptr)
            NCCLCHECK(ncclCommDeregister(comm, meta->handle));
          NCCLCHECK(ncclMemFree(*buff));
          *buff = nullptr;
          meta->handle = nullptr;
      }
      NCCLCHECK(ncclMemAlloc(buff, expectBytes + COMPBUFF_EXCESS_SIZE_T));
      if(comm != nullptr)
        NCCLCHECK(ncclCommRegister(comm, *buff, expectBytes, &(meta->handle)));
      if(tempbuff != nullptr){
        CUDACHECK(cudaMemcpy(*buff, tempbuff, meta->maxBytes, cudaMemcpyDeviceToDevice));
        CUDACHECK(cudaFree(tempbuff));
      }
      meta->maxBytes = expectBytes;
      CUDACHECK(cudaDeviceSynchronize());
    }
    return ncclSuccess;
}

ncclResult_t cocclBuffAlloc(void** buff, size_t expectBytes, ncclComm_t comm){
    NCCLCHECK(allocAndRegBuff(buff, expectBytes, comm));
    // *buff = cbuff;
    return ncclSuccess;
}

