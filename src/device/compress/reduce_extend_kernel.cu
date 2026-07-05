
#include "nccl.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <cmath>
#include <limits>
#include "reduce_extend_kernel.h"


template<typename T>
__device__ inline void __store_float(T * array, float data) {
    array[0] = data;
}

template<>
__device__ inline void __store_float<half>(half * array, float data) {
    array[0] = __float2half(data);
}

template<typename T>
__device__ inline T getInfinity();

template<>
__device__ inline float getInfinity<float>(){
    return INFINITY;
}

template<>
__device__ inline half getInfinity<half>(){
    return __float2half(INFINITY);
}

__device__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
    assumed = old;
    old = ::atomicCAS(address_as_i, assumed,
    __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    
    do {
    assumed = old;
    old = ::atomicCAS(address_as_i, assumed,
    __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

template<typename T>
__device__ void warpReduce(volatile T* smin, volatile T* smax, const int tid, const int blockSize){
    if(blockSize >=64) {
        smin[tid] = smin[tid] < smin[tid + 32] ? smin[tid] : smin[tid + 32];
        smax[tid] = smax[tid] > smax[tid + 32] ? smax[tid] : smax[tid + 32];
    }
    if(blockSize >=32) {
        smin[tid] = smin[tid] < smin[tid + 16] ? smin[tid] : smin[tid + 16];
        smax[tid] = smax[tid] > smax[tid + 16] ? smax[tid] : smax[tid + 16];
    }
    if(blockSize >=16) {
        smin[tid] = smin[tid] < smin[tid + 8] ? smin[tid] : smin[tid + 8];
        smax[tid] = smax[tid] > smax[tid + 8] ? smax[tid] : smax[tid + 8];
    }
    if(blockSize >=8) {
        smin[tid] = smin[tid] < smin[tid + 4] ? smin[tid] : smin[tid + 4];
        smax[tid] = smax[tid] > smax[tid + 4] ? smax[tid] : smax[tid + 4];
    }
    if(blockSize >=4) {
        smin[tid] = smin[tid] < smin[tid + 2] ? smin[tid] : smin[tid + 2];
        smax[tid] = smax[tid] > smax[tid + 2] ? smax[tid] : smax[tid + 2];
    }
    if(blockSize >=2) {
        smin[tid] = smin[tid] < smin[tid + 1] ? smin[tid] : smin[tid + 1];
        smax[tid] = smax[tid] > smax[tid + 1] ? smax[tid] : smax[tid + 1];
    }
}

template<typename T>
__device__ void warpShuffleMinMax(T &localMin, T &localMax, unsigned int mask, int blockSize){
    if(blockSize >= 32){
        T tMax = __shfl_down_sync(mask, localMax, 16);
        T tMin = __shfl_down_sync(mask, localMin, 16);
        localMin = localMin < tMin ? localMin : tMin;
        localMax = localMax > tMax ? localMax : tMax;
    }
    if(blockSize >= 16){
        T tMin = __shfl_down_sync(mask, localMin, 8);
        T tMax = __shfl_down_sync(mask, localMax, 8);
        localMin = localMin < tMin ? localMin : tMin;
        localMax = localMax > tMax ? localMax : tMax;
    }
    if(blockSize >= 8){
        T tMin = __shfl_down_sync(mask, localMin, 4);
        T tMax = __shfl_down_sync(mask, localMax, 4);
        localMin = localMin < tMin ? localMin : tMin;
        localMax = localMax > tMax ? localMax : tMax;
    } 
    if(blockSize >= 4){
        T tMin = __shfl_down_sync(mask, localMin, 2);
        T tMax = __shfl_down_sync(mask, localMax, 2);
        localMin = localMin < tMin ? localMin : tMin;
        localMax = localMax > tMax ? localMax : tMax;
    } 
    if(blockSize >= 2){
        T tMin = __shfl_down_sync(mask, localMin, 1);
        T tMax = __shfl_down_sync(mask, localMax, 1);
        localMin = localMin < tMin ? localMin : tMin;
        localMax = localMax > tMax ? localMax : tMax;
    } 
}

template<typename T, size_t blockSize>
__global__ void 
maxMinBlockReduce(const void* input, const size_t chunkCount, void* output, const size_t compChunkCount){
    // extern __shared__  unsigned char smem[];
    __shared__ T smem[2 * blockSize];


    // T* sharedMem = reinterpret_cast<T*>(smem);
    T* sharedMem = smem;


    T* inputbuff = (T*)input;
    uint8_t* outputbuff = (uint8_t*)output;
    
    T* sharedMin = sharedMem;
    T* sharedMax = sharedMem + blockDim.x;
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    // const int blockSize = blockDim.x;


    T localMin = getInfinity<T>();
    T localMax = -getInfinity<T>();
    while(idx < chunkCount){
        int k = idy * chunkCount;
        localMax = (localMax > inputbuff[k + idx]) ? localMax : inputbuff[k + idx];
        localMin = (localMin < inputbuff[k + idx]) ? localMin : inputbuff[k + idx];
        idx += blockDim.x * gridDim.x;
    }

    sharedMax[tid] = localMax;
    sharedMin[tid] = localMin;

    __syncthreads();

    if(blockSize >= 1024){
        if(tid < 512){
            sharedMax[tid] = localMax = (localMax > sharedMax[tid + 512]) ? localMax : sharedMax[tid + 512];
            sharedMin[tid] = localMin = (localMin < sharedMin[tid + 512]) ? localMin : sharedMin[tid + 512];
        }
        __syncthreads();
    }

    if(blockSize >= 512){
        if(tid < 256){
            sharedMax[tid] = localMax = (localMax > sharedMax[tid + 256]) ? localMax : sharedMax[tid + 256];
            sharedMin[tid] = localMin = (localMin < sharedMin[tid + 256]) ? localMin : sharedMin[tid + 256];
        }
        __syncthreads();
    }

    if(blockSize >= 256){
        if(tid < 128){
            sharedMax[tid] = localMax = (localMax > sharedMax[tid + 128]) ? localMax : sharedMax[tid + 128];
            sharedMin[tid] = localMin = (localMin < sharedMin[tid + 128]) ? localMin : sharedMin[tid + 128];
        }
        __syncthreads();
    }

    if(blockSize >= 128){
        if(tid < 64){
            sharedMax[tid] = localMax = (localMax > sharedMax[tid + 64]) ? localMax : sharedMax[tid + 64];
            sharedMin[tid] = localMin = (localMin < sharedMin[tid + 64]) ? localMin : sharedMin[tid + 64];
        }
        __syncthreads();
    }

    // if(blockSize >= 64){
    //     if(tid < 32){
    //         sharedMax[tid] = localMax = (localMax > sharedMax[tid + 32]) ? localMax : sharedMax[tid + 32];
    //         sharedMin[tid] = localMin = (localMin < sharedMin[tid + 32]) ? localMin : sharedMin[tid + 32];
    //     }
    //     __syncthreads();
    // }
    // if(tid < 32){
    //     warpShuffleMinMax<T>(localMin, localMax, 0xffffffff, blockSize);
    // }

    if(tid < 32){
        warpReduce<T>(sharedMin, sharedMax, tid, blockSize);
    }

    if(tid == 0){

        // T localMax = sharedMax[0];
        // T localMin = sharedMin[0];
        // warpShuffleMinMax<T,blockDim.x>(localMin, localMax, 0xffffffff);
        T* min_ = reinterpret_cast<T *>(outputbuff + idy * compChunkCount);
        T* max_ = reinterpret_cast<T *>(outputbuff + idy * compChunkCount + sizeof(T));
        // atomicMin(min_, localMin);
        // atomicMax(max_, localMax);
        atomicMin(min_, sharedMin[0]);
        atomicMax(max_, sharedMax[0]);
    }
}

template <typename T>
__global__ void 
reduceColl(const void *input1, const void *input2, void *output, size_t inputCount) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    T* inputbuff1 = (T*)input1;
    T* inputbuff2 = (T*)input2;
    T* outputbuff = (T*)output;

    if (idx < inputCount) {
        outputbuff[idx] = inputbuff1[idx] + inputbuff2[idx];
    }
}

template <typename T>
__global__ void InitMinMax(void* input, const size_t chunkCount, const size_t numChunk){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint8_t* inputbuff = (uint8_t*) input;
    if(idx < numChunk){
        __store_float(reinterpret_cast<T *>(inputbuff + idx * chunkCount), getInfinity<T>());
        __store_float(reinterpret_cast<T *>(inputbuff + idx * chunkCount + sizeof(T)), -getInfinity<T>());
    }
}

template<typename TO, typename FROM>
__device__ __forceinline__ TO type_cast(FROM val){
    return type_cast(val);
}

template<>
__device__ __forceinline__ float type_cast(float val){
    return val;
}

template<>
__device__ __forceinline__ float type_cast(__half val){
    return __half2float(val);
}

template<>
__device__ __forceinline__ float type_cast(__nv_bfloat16 val){
    return __bfloat162float(val);
}


template<>
__device__ __forceinline__ __half type_cast(float val){
    return __float2half(val);
}

template<>
__device__ __forceinline__ __half type_cast(__half val){
    return val;
}

template<>
__device__ __forceinline__ __nv_bfloat16 type_cast(float val){
    return __float2bfloat16(val);
}

template<>
__device__ __forceinline__ __nv_bfloat16 type_cast(__nv_bfloat16 val){
    return val;
}



template<typename T, unsigned int blockDimY>
__device__ void
block_y_reduce(volatile float sdata[][blockDimY], unsigned int tidx, unsigned int tidy) {
    if (blockDimY >= 32) {
        if (tidy < 16) { sdata[tidx][tidy] = sdata[tidx][tidy] + sdata[tidx][tidy + 16]; }
        __syncthreads();
    }
    if (blockDimY >= 16) {
        if (tidy < 8) { sdata[tidx][tidy] = sdata[tidx][tidy] + sdata[tidx][tidy + 8]; }
        __syncthreads();
    }
    if (blockDimY >= 8) {
        if (tidy < 4) { sdata[tidx][tidy] = sdata[tidx][tidy] + sdata[tidx][tidy + 4]; }
        __syncthreads();
    }
    if (blockDimY >= 4) {
        if (tidy < 2) { sdata[tidx][tidy] = sdata[tidx][tidy] + sdata[tidx][tidy + 2]; }
        __syncthreads();
    }
    if (blockDimY >= 2) {
        if (tidy < 1) { sdata[tidx][tidy] = sdata[tidx][tidy] + sdata[tidx][tidy + 1]; }
        __syncthreads();
    }
}

template<unsigned int blockDimX, unsigned int blockDimY, typename T>
__global__ void reduceChunk(const void *input, int chunkCount, int numChunks, void* output) {
    __shared__ float sdata[blockDimX][blockDimY];
    T* inputbuff = (T*)input;
    T* outputbuff = (T*)output;

    unsigned int tidx = threadIdx.x;
    unsigned int tidy = threadIdx.y;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // load to shared memory
    float sum = 0.0;
    for (int i = idy; i < numChunks && idx < chunkCount; i += blockDim.y) {
        sum = sum + type_cast<float>(inputbuff[chunkCount * i + idx]);
    }

    sdata[tidx][tidy] = sum;
    __syncthreads();

    block_y_reduce<T, blockDimY>(sdata, tidx, tidy);

    // write to global memory
    if (tidy == 0 && idx < chunkCount) {
        outputbuff[idx] = type_cast<T>(sdata[tidx][tidy]);
    }
}

void minMaxReduction(const void* input, const size_t chunkCount, void* output, const size_t outputChunkCount, 
    const size_t numChunks, ncclDataType_t datatype, cudaStream_t stream){
    int InitBlock = numChunks < 1024? numChunks: 1024;
    int InitGrid = DIVUP(numChunks, InitBlock);

    int block = chunkCount < 1024 ? chunkCount : 1024;
    // dim3 grid(DIVUP(chunkCount, 32 * block), numChunks);
    // dim3 grid(1024, numChunks);

    dim3 grid(128, numChunks);

    if(datatype == ncclDataType_t::ncclFloat16){
        
    } else if(datatype == ncclDataType_t::ncclFloat32){
        InitMinMax<float> <<<InitGrid, InitBlock, 0, stream>>> (output, outputChunkCount, numChunks);
        maxMinBlockReduce<float, 1024> <<<grid, block, 0, stream>>> (input, chunkCount, output, outputChunkCount);
    }
}


extern "C"{

ncclResult_t launchReductionColl(const void* input1, const void* input2, void* output, ncclDataType_t datatype, size_t inputCount, 
    cudaStream_t stream){
    int block = inputCount < 1024 ? inputCount : 1024;
    dim3 grid = DIVUP(inputCount, block);
    if(datatype == ncclDataType_t::ncclFloat16){
        reduceColl<half> <<<grid, block, 0, stream>>>(input1, input2, output, inputCount);
    } else if(datatype == ncclDataType_t::ncclFloat32){
        reduceColl<float> <<<grid, block, 0, stream>>>(input1, input2, output, inputCount);
    }
    CUDACHECK(cudaGetLastError());
    return ncclSuccess;
}

ncclResult_t launchReduceChunk(const void* input, size_t chunkCount, void* output, ncclDataType_t datatype, int numChunks, 
    cudaStream_t stream){
    if(datatype == ncclDataType_t::ncclFloat32){
        if (numChunks <= 4) {
            dim3 grid(DIVUP(chunkCount, 512), 1);
            dim3 block(512, 2);
            reduceChunk<512, 2, float><<<grid, block, 0, stream>>>(input, chunkCount, numChunks, output);
        } else if (numChunks <= 8) {
            dim3 grid(DIVUP(chunkCount, 256), 1);
            dim3 block(256, 4);
            reduceChunk<256, 4, float><<<grid, block, 0, stream>>>(input, chunkCount, numChunks, output);
        } else if (numChunks <= 16) {
            dim3 grid(DIVUP(chunkCount, 128), 1);
            dim3 block(128, 8);
            reduceChunk<128, 8, float><<<grid, block, 0, stream>>>(input, chunkCount, numChunks, output);
        } else if (numChunks <= 32) {
            dim3 grid(DIVUP(chunkCount, 64), 1);
            dim3 block(64, 16);
            reduceChunk<64, 16, float><<<grid, block, 0, stream>>>(input, chunkCount, numChunks, output);
        } else {
            dim3 grid(DIVUP(chunkCount, 32), 1);
            dim3 block(32, 32);
            reduceChunk<32, 32, float><<<grid, block, 0, stream>>>(input, chunkCount, numChunks, output);
        }
    } else if(datatype == ncclDataType_t::ncclFloat16){
        if (numChunks <= 4) {
            dim3 grid(DIVUP(chunkCount, 512), 1);
            dim3 block(512, 2);
            reduceChunk<512, 2, __half><<<grid, block, 0, stream>>>(input, chunkCount, numChunks, output);
        } else if (numChunks <= 8) {
            dim3 grid(DIVUP(chunkCount, 256), 1);
            dim3 block(256, 4);
            reduceChunk<256, 4, __half><<<grid, block, 0, stream>>>(input, chunkCount, numChunks, output);
        } else if (numChunks <= 16) {
            dim3 grid(DIVUP(chunkCount, 128), 1);
            dim3 block(128, 8);
            reduceChunk<128, 8, __half><<<grid, block, 0, stream>>>(input, chunkCount, numChunks, output);
        } else if (numChunks <= 32) {
            dim3 grid(DIVUP(chunkCount, 64), 1);
            dim3 block(64, 16);
            reduceChunk<64, 16, __half><<<grid, block, 0, stream>>>(input, chunkCount, numChunks, output);
        } else {
            dim3 grid(DIVUP(chunkCount, 32), 1);
            dim3 block(32, 32);
            reduceChunk<32, 32, __half><<<grid, block, 0, stream>>>(input, chunkCount, numChunks, output);
        }
    } else if(datatype == ncclDataType_t::ncclBfloat16){
        if (numChunks <= 4) {
            dim3 grid(DIVUP(chunkCount, 512), 1);
            dim3 block(512, 2);
            reduceChunk<512, 2, __nv_bfloat16><<<grid, block, 0, stream>>>(input, chunkCount, numChunks, output);
        } else if (numChunks <= 8) {
            dim3 grid(DIVUP(chunkCount, 256), 1);
            dim3 block(256, 4);
            reduceChunk<256, 4, __nv_bfloat16><<<grid, block, 0, stream>>>(input, chunkCount, numChunks, output);
        } else if (numChunks <= 16) {
            dim3 grid(DIVUP(chunkCount, 128), 1);
            dim3 block(128, 8);
            reduceChunk<128, 8, __nv_bfloat16><<<grid, block, 0, stream>>>(input, chunkCount, numChunks, output);
        } else if (numChunks <= 32) {
            dim3 grid(DIVUP(chunkCount, 64), 1);
            dim3 block(64, 16);
            reduceChunk<64, 16, __nv_bfloat16><<<grid, block, 0, stream>>>(input, chunkCount, numChunks, output);
        } else {
            dim3 grid(DIVUP(chunkCount, 32), 1);
            dim3 block(32, 32);
            reduceChunk<32, 32, __nv_bfloat16><<<grid, block, 0, stream>>>(input, chunkCount, numChunks, output);
        }
    }
    CUDACHECK(cudaGetLastError());
    return ncclSuccess;
}

}
