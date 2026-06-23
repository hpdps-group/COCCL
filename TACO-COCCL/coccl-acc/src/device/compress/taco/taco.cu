#include "taco.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include "nccl.h"
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#ifndef HAS_CUDA_MALLOC_ASYNC
#if CUDART_VERSION >= 11020
#define HAS_CUDA_MALLOC_ASYNC 1
#else
#define HAS_CUDA_MALLOC_ASYNC 0
#endif
#endif

inline cudaError_t safeCudaMallocAsync(void** ptr, size_t bytes, cudaStream_t stream) {
#if HAS_CUDA_MALLOC_ASYNC
    return cudaMallocAsync(ptr, bytes, stream);
#else
    return cudaMalloc(ptr, bytes);
#endif
}

#define __hidden __attribute__ ((visibility("hidden")))
#define DIVUP(x,y) (((x)+(y)-1)/(y))

inline size_t getTypeSize(ncclDataType_t type) {
    switch (type) {
        case ncclInt8:     return 1;
        case ncclUint8:    return 1;
        case ncclInt32:    return 4;
        case ncclUint32:   return 4;
        case ncclInt64:    return 8;
        case ncclUint64:   return 8;
        case ncclFloat16:  return 2;
        case ncclFloat32:  return 4;
        case ncclFloat64:  return 8;
        #if defined(__CUDA_BF16_TYPES_EXIST__)
        case ncclBfloat16: return 2;
        #endif
        default:           return 1;
    }
}

struct fp8Config {
    int fp8_format;
    int saturation;
    int use_scale;
    int group_size;
    float target_range;
    int safe_mode;
    int use_as_hadamard; // Combined parameter
    float lambda;
    float fp8_max_val;
    int pivotSwap;
};

__hidden void parseFp8Config(const char* configFile, void** compConfig, int nodes, int devicesPerNodes) {
    *compConfig = (void*) malloc(sizeof(fp8Config));
    fp8Config* config = reinterpret_cast<fp8Config*>(*compConfig);

    config->fp8_format = 0;
    config->saturation = 1;
    config->use_scale = 1;
    config->group_size = 128;
    config->target_range = 448.0f;
    config->safe_mode = 0;
    config->use_as_hadamard = 1;
    config->lambda = 1e-6f;
    config->fp8_max_val = 448.0f;
    config->pivotSwap = 0;

    if (!configFile) return;

    std::pair<const char*, const char*>* configPairs = nullptr;
    int configPairCount = 0;

    loadConfigPair(configFile, &configPairs, &configPairCount);
    if (configPairs == nullptr) return;

    for (int i = 0; i < configPairCount; i++) {
        const char* key = configPairs[i].first;
        const char* val = configPairs[i].second;
        if (strcmp(key, "fp8_type") == 0 || strcmp(key, "fp8_format") == 0) {
            config->fp8_format = (strcmp(val, "E5M2") == 0) ? 1 : 0;
            config->fp8_max_val = (config->fp8_format == 1) ? 57344.0f : 448.0f;
        } else if (strcmp(key, "saturate") == 0 || strcmp(key, "saturation") == 0) {
            config->saturation = atoi(val);
        } else if (strcmp(key, "use_scale") == 0) {
            config->use_scale = atoi(val);
        } else if (strcmp(key, "group_size") == 0) {
            config->group_size = atoi(val);
        } else if (strcmp(key, "target_range") == 0) {
            config->target_range = atof(val);
        } else if (strcmp(key, "safe_mode") == 0) {
            config->safe_mode = atoi(val);
        } else if (strcmp(key, "use_as_hadamard") == 0 || strcmp(key, "as_hadamard") == 0) {
            config->use_as_hadamard = atoi(val);
        } else if (strcmp(key, "lambda") == 0) {
            config->lambda = atof(val);
        } else if (strcmp(key, "fp8_max_val") == 0) {
            config->fp8_max_val = atof(val);
        } else if (strcmp(key, "pivotSwap") == 0) {
            config->pivotSwap = atoi(val);
        }
    }
}

__device__ __forceinline__ float safe_divide(float a, float b) {
    return (fabsf(b) < 1e-12f) ? 0.0f : (a / b);
}

__device__ __forceinline__ float clamp_to_fp8_range(float x, float max_val) {
    return fmaxf(fminf(x, max_val), -max_val);
}

__device__ __forceinline__ float check_and_fix_nan_inf(float x) {
    return (isnan(x) || isinf(x)) ? 0.0f : x;
}

__device__ __host__ __forceinline__ float clampf(float x, float min_val, float max_val) {
    return fmaxf(fminf(x, max_val), min_val);
}

__global__ void bf16_to_float_kernel(const __nv_bfloat16* in, float* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
}

__global__ void float_to_bf16_kernel(const float* in, __nv_bfloat16* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in[i]);
}

__global__ void hadamard_groups_float(float* data, int orgChunkCount, int group_size, int groups_per_chunk) {
    int global_block = blockIdx.x;
    int chunk_id = global_block / groups_per_chunk;
    int group_id = global_block % groups_per_chunk;

    int start = chunk_id * orgChunkCount + group_id * group_size;
    int max_len = orgChunkCount - group_id * group_size;
    int len = group_size;
    if (len > max_len) len = max_len;
    if (len <= 0) return;

    int n = 1;
    while ((n << 1) <= len) n <<= 1;
    if (n == 1) return;

    extern __shared__ float s[];
    int tid = threadIdx.x;

    for (int i = tid; i < n; i += blockDim.x) {
        s[i] = data[start + i];
    }
    __syncthreads();

    for (int stride = 1; stride < n; stride <<= 1) {
        for (int idx = tid; idx < n; idx += blockDim.x) {
            int base = (idx / (2 * stride)) * (2 * stride);
            int offset = idx % (2 * stride);
            if (offset < stride) {
                int i = base + offset;
                int j = i + stride;
                if (j < n) {
                    float a = s[i];
                    float b = s[j];
                    s[i] = a + b;
                    s[j] = a - b;
                }
            }
        }
        __syncthreads();
    }

    float inv_sqrt_n = rsqrtf((float)n);
    for (int i = tid; i < n; i += blockDim.x) {
        s[i] *= inv_sqrt_n;
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        data[start + i] = s[i];
    }
}

__global__ void compute_group_var(
    const float* input,
    float* vars,
    int orgChunkCount,
    int numChunks,
    int group_size,
    float lambda)
{
    int global_block = blockIdx.x;
    int groups_per_chunk = DIVUP(orgChunkCount, group_size);
    int chunk_id = global_block / groups_per_chunk;
    int group_id = global_block % groups_per_chunk;

    int chunk_start = chunk_id * orgChunkCount;
    int group_start = chunk_start + group_id * group_size;
    int group_end = min(group_start + group_size, chunk_start + orgChunkCount);
    int group_len = group_end - group_start;
    if (group_len <= 0) {
        vars[global_block] = lambda;
        return;
    }

    float sum_sq = 0.0f;
    for (int i = group_start + threadIdx.x; i < group_end; i += blockDim.x) {
        float x = input[i];
        sum_sq += x * x;
    }

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    sdata[tid] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        float var = (sdata[0] / (float)group_len) + lambda;
        vars[global_block] = var;
    }
}

__global__ void compute_adaptive_scales(
    const float* vars,
    float* scales,
    int groups_per_chunk,
    int numChunks,
    float fp8_max_val)
{
    int global_block = blockIdx.x;
    float var = vars[global_block];
    if (var < 1e-12f) var = 1e-12f;

    float scale = fp8_max_val / sqrtf(var);
    scale = clampf(scale, 1e-3f, 1e3f);
    scales[global_block] = scale;
}

__global__ void apply_adaptive_scale(
    float* input,
    const float* scales,
    int orgChunkCount,
    int numChunks,
    int group_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_count = orgChunkCount * numChunks;
    if (idx >= total_count) return;

    int chunk_id = idx / orgChunkCount;
    int in_chunk_idx = idx % orgChunkCount;
    int groups_per_chunk = DIVUP(orgChunkCount, group_size);
    int group_id = in_chunk_idx / group_size;
    int global_group_id = chunk_id * groups_per_chunk + group_id;

    input[idx] *= scales[global_group_id];
}

__global__ void apply_adaptive_inv_scale(
    float* input,
    const float* scales,
    int decompChunkCount,
    int numChunks,
    int group_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_count = decompChunkCount * numChunks;
    if (idx >= total_count) return;

    int chunk_id = idx / decompChunkCount;
    int in_chunk_idx = idx % decompChunkCount;
    int groups_per_chunk = DIVUP(decompChunkCount, group_size);
    int group_id = in_chunk_idx / group_size;
    int global_group_id = chunk_id * groups_per_chunk + group_id;

    float scale = scales[global_group_id];
    if (scale < 1e-12f) scale = 1e-12f;
    input[idx] /= scale;
}

__global__ void compute_scales_group_per_chunk(
    const float* input,
    float* scales,
    int orgChunkCount,
    int groups_per_chunk,
    int group_size,
    float target_range)
{
    int group_id = blockIdx.x;
    int start = group_id * group_size;
    int end = min((group_id + 1) * group_size, orgChunkCount);

    float local_max = 0.0f;
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        float v = fabsf(input[i]);
        if (isnan(v) || isinf(v)) v = 0.0f;
        if (v > local_max) local_max = v;
    }

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && sdata[tid + stride] > sdata[tid]) {
            sdata[tid] = sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float max_val = sdata[0];
        if (max_val < 1e-12f) max_val = 1e-12f;
        float scale = safe_divide(max_val, target_range);
        scale = clampf(scale, 1e-12f, 1e6f);
        scales[group_id] = scale;
    }
}

__global__ void compute_scales_group_per_chunk_bf16(
    const __nv_bfloat16* input,
    float* scales,
    int orgChunkCount,
    int groups_per_chunk,
    int group_size,
    float target_range)
{
    int group_id = blockIdx.x;
    int start = group_id * group_size;
    int end = min((group_id + 1) * group_size, orgChunkCount);

    float local_max = 0.0f;
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        float v = fabsf(__bfloat162float(input[i]));
        if (isnan(v) || isinf(v)) v = 0.0f;
        if (v > local_max) local_max = v;
    }

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && sdata[tid + stride] > sdata[tid]) {
            sdata[tid] = sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float max_val = sdata[0];
        if (max_val < 1e-12f) max_val = 1e-12f;
        float scale = safe_divide(max_val, target_range);
        scale = clampf(scale, 1e-12f, 1e6f);
        scales[group_id] = scale;
    }
}

__global__ void compress_fp32_to_fp8_group_per_chunk(
    const float* input,
    __nv_fp8_storage_t* output_fp8,
    const float* scales,
    int orgChunkCount,
    int groups_per_chunk,
    int group_size,
    __nv_saturation_t saturate,
    __nv_fp8_interpretation_t fp8_type)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= orgChunkCount) return;

    int group_id = idx / group_size;
    float scale = scales[group_id];
    
    float input_val = input[idx];
    input_val = check_and_fix_nan_inf(input_val);
    float scaled = safe_divide(input_val, scale);
    scaled = clamp_to_fp8_range(scaled, 448.0f);

    output_fp8[idx] = __nv_cvt_float_to_fp8(scaled, saturate, fp8_type);
}

__global__ void compress_bf16_to_fp8_group_per_chunk(
    const __nv_bfloat16* input,
    __nv_fp8_storage_t* output_fp8,
    const float* scales,
    int orgChunkCount,
    int groups_per_chunk,
    int group_size,
    __nv_saturation_t saturate,
    __nv_fp8_interpretation_t fp8_type)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= orgChunkCount) return;

    int group_id = idx / group_size;
    float scale = scales[group_id];
    
    float input_val = __bfloat162float(input[idx]);
    input_val = check_and_fix_nan_inf(input_val);
    float scaled = safe_divide(input_val, scale);
    scaled = clamp_to_fp8_range(scaled, 448.0f);

    output_fp8[idx] = __nv_cvt_float_to_fp8(scaled, saturate, fp8_type);
}

__global__ void decompress_fp8_to_fp32_group_per_chunk(
    const __nv_fp8_storage_t* input_fp8,
    float* output,
    const float* scales,
    int orgChunkCount,
    int groups_per_chunk,
    int group_size,
    __nv_fp8_interpretation_t fp8_type)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= orgChunkCount) return;

    int group_id = idx / group_size;
    float scale = scales[group_id];

    __nv_fp8_storage_t s = input_fp8[idx];
    __half_raw hr = __nv_cvt_fp8_to_halfraw(s, fp8_type);
    __half h = *reinterpret_cast<__half*>(&hr);
    float val = __half2float(h);
    
    float result = val * scale;
    result = check_and_fix_nan_inf(result);
    output[idx] = result;
}

__global__ void decompress_fp8_to_bf16_group_per_chunk(
    const __nv_fp8_storage_t* input_fp8,
    __nv_bfloat16* output,
    const float* scales,
    int orgChunkCount,
    int groups_per_chunk,
    int group_size,
    __nv_fp8_interpretation_t fp8_type)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= orgChunkCount) return;

    int group_id = idx / group_size;
    float scale = scales[group_id];

    __nv_fp8_storage_t s = input_fp8[idx];
    __half_raw hr = __nv_cvt_fp8_to_halfraw(s, fp8_type);
    __half h = *reinterpret_cast<__half*>(&hr);
    float val = __half2float(h);
    
    float result = val * scale;
    result = check_and_fix_nan_inf(result);
    output[idx] = __float2bfloat16(result);
}

template<typename T>
__global__ void compress_to_fp8_kernel(const void* input, __nv_fp8_storage_t* output,
                                       int chunkCount, __nv_saturation_t saturate,
                                       __nv_fp8_interpretation_t fp8_type)
{
    const T* in = reinterpret_cast<const T*>(input);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chunkCount) return;
    
    float v;
    if (sizeof(T) == 2) {
        if (std::is_same<T, __half>::value) {
            v = __half2float(reinterpret_cast<const __half*>(in)[idx]);
        } else {
            v = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(in)[idx]);
        }
    } else {
        v = reinterpret_cast<const float*>(in)[idx];
    }
    v = check_and_fix_nan_inf(v);
    v = clamp_to_fp8_range(v, 448.0f);
    output[idx] = __nv_cvt_float_to_fp8(v, saturate, fp8_type);
}

template<typename T>
__global__ void decompress_from_fp8_kernel(const __nv_fp8_storage_t* input, void* output,
                                           int chunkCount, __nv_fp8_interpretation_t fp8_type)
{
    T* out = reinterpret_cast<T*>(output);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chunkCount) return;
    
    __nv_fp8_storage_t s = input[idx];
    __half_raw hr = __nv_cvt_fp8_to_halfraw(s, fp8_type);
    __half h = *reinterpret_cast<__half*>(&hr);
    float f = __half2float(h);
    f = check_and_fix_nan_inf(f);
    
    if (sizeof(T) == 2) {
        if (std::is_same<T, __half>::value) {
            reinterpret_cast<__half*>(out)[idx] = __float2half(f);
        } else {
            reinterpret_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16(f);
        }
    } else {
        reinterpret_cast<float*>(out)[idx] = f;
    }
}

__hidden cudaError_t fallbackNoCompression(const void* orgbuff, void** compbuff,
                                          size_t orgChunkCount, ncclDataType_t orgDatatype,
                                          size_t* compChunkCount, ncclDataType_t* compDatatype,
                                          size_t numChunks, cudaStream_t stream) {
    *compDatatype = orgDatatype;
    *compChunkCount = orgChunkCount;
    size_t typeSize = getTypeSize(orgDatatype);
    size_t totalBytes = orgChunkCount * typeSize * numChunks;
    if (*compbuff == nullptr) {
        cudaError_t err = safeCudaMallocAsync((void**)compbuff, totalBytes, stream);
        if (err != cudaSuccess) return err;
    }
    return cudaMemcpyAsync(*compbuff, orgbuff, totalBytes, cudaMemcpyDeviceToDevice, stream);
}

__hidden cudaError_t launchCompress(const void* orgbuff, void** compbuff,
                                    size_t orgChunkCount, ncclDataType_t orgDatatype,
                                    size_t* compChunkCount, ncclDataType_t* compDatatype,
                                    size_t numChunks, void* config,
                                    cudaMemPool_t compMemPool, cudaStream_t stream) {
    static bool is_printed = false;
    if (!is_printed) {
        printf("\nTACO Plugin (AS-Hadamard Merged) Loaded.\n");
        is_printed = true;
    }

    if (orgChunkCount == 0 || numChunks == 0) return cudaSuccess;
    nvtxRangePushA(">>> MY_ASH_ALGO <<<");

    fp8Config cfg = config ? *reinterpret_cast<fp8Config*>(config) : fp8Config();
    if (!config) {
        cfg.fp8_format = 0; cfg.saturation = 1; cfg.use_scale = 1;
        cfg.group_size = 128; cfg.target_range = 448.0f; cfg.safe_mode = 0;
        cfg.use_as_hadamard = 0;
        cfg.pivotSwap = 0; cfg.lambda = 1e-6f; cfg.fp8_max_val = 448.0f;
    }

    if (cfg.safe_mode) return fallbackNoCompression(orgbuff, compbuff, orgChunkCount, orgDatatype, compChunkCount, compDatatype, numChunks, stream);
    if (orgDatatype != ncclFloat32 && orgDatatype != ncclFloat16 && orgDatatype != ncclBfloat16) return fallbackNoCompression(orgbuff, compbuff, orgChunkCount, orgDatatype, compChunkCount, compDatatype, numChunks, stream);

    *compDatatype = ncclUint8;
    __nv_fp8_interpretation_t fp8_type = (cfg.fp8_format == 0) ? __NV_E4M3 : __NV_E5M2;
    __nv_saturation_t saturate = (cfg.saturation == 1) ? __NV_SATFINITE : __NV_NOSAT;

    int groups_per_chunk = 0;
    size_t bytes_per_chunk = orgChunkCount;
    size_t adaptive_scale_bytes_per_chunk = 0;

    if (cfg.use_scale && cfg.group_size > 0) {
        groups_per_chunk = DIVUP((int)orgChunkCount, cfg.group_size);
        bytes_per_chunk += groups_per_chunk * sizeof(float);
    }
    
    if (cfg.use_as_hadamard) {
        if (groups_per_chunk == 0) groups_per_chunk = DIVUP((int)orgChunkCount, cfg.group_size);
        adaptive_scale_bytes_per_chunk = (size_t)groups_per_chunk * sizeof(float);
        bytes_per_chunk += adaptive_scale_bytes_per_chunk;
    }
    *compChunkCount = bytes_per_chunk;
    size_t totalBytes = bytes_per_chunk * numChunks;

    if (*compbuff == nullptr) {
        cudaError_t err = safeCudaMallocAsync((void**)compbuff, totalBytes, stream);
        if (err != cudaSuccess) return err;
    }

    cudaError_t kernelErr = cudaSuccess;
    float* hadamard_input = nullptr;
    bool allocated_temp = false;
    float* d_adaptive_scales = nullptr;

    if (cfg.use_as_hadamard) {
        if (cfg.pivotSwap) return cudaErrorNotSupported;

        size_t totalCount = orgChunkCount * numChunks;
        size_t tempBytes = totalCount * sizeof(float);
        kernelErr = cudaMalloc(&hadamard_input, tempBytes);
        if (kernelErr != cudaSuccess) return kernelErr;
        allocated_temp = true;

        if (orgDatatype == ncclFloat32) {
            kernelErr = cudaMemcpyAsync(hadamard_input, orgbuff, tempBytes, cudaMemcpyDeviceToDevice, stream);
        } else if (orgDatatype == ncclBfloat16) {
            int block = 256;
            int grid = DIVUP((int)totalCount, block);
            bf16_to_float_kernel<<<grid, block, 0, stream>>>((const __nv_bfloat16*)orgbuff, hadamard_input, totalCount);
        } else {
            cudaFree(hadamard_input);
            return cudaErrorNotSupported;
        }
        kernelErr = cudaGetLastError();
        if (kernelErr != cudaSuccess) {
            if (allocated_temp) cudaFree(hadamard_input);
            return kernelErr;
        }

        int groups_total = (int)numChunks * groups_per_chunk;
        size_t adaptive_scale_total_bytes = (size_t)groups_total * sizeof(float);

        float* d_vars = nullptr;
        kernelErr = cudaMalloc(&d_vars, adaptive_scale_total_bytes);
        if (kernelErr != cudaSuccess) { if (allocated_temp) cudaFree(hadamard_input); return kernelErr; }
        
        kernelErr = cudaMalloc(&d_adaptive_scales, adaptive_scale_total_bytes);
        if (kernelErr != cudaSuccess) { cudaFree(d_vars); if (allocated_temp) cudaFree(hadamard_input); return kernelErr; }

        int var_threads = 256;
        int var_grid = groups_total;
        size_t var_shmem = var_threads * sizeof(float);
        compute_group_var<<<var_grid, var_threads, var_shmem, stream>>>(
            hadamard_input, d_vars, (int)orgChunkCount, (int)numChunks, cfg.group_size, cfg.lambda);

        compute_adaptive_scales<<<var_grid, var_threads, 0, stream>>>(
            d_vars, d_adaptive_scales, groups_per_chunk, (int)numChunks, cfg.fp8_max_val);

        int apply_threads = 256;
        int apply_grid = DIVUP((int)totalCount, apply_threads);
        apply_adaptive_scale<<<apply_grid, apply_threads, 0, stream>>>(
            hadamard_input, d_adaptive_scales, (int)orgChunkCount, (int)numChunks, cfg.group_size);

        cudaFree(d_vars);

        int grid = (int)numChunks * groups_per_chunk;
        int block = (cfg.group_size <= 1024) ? cfg.group_size : 1024;
        block = std::max(block, 32);
        size_t shmem = ((cfg.group_size <= 1024) ? cfg.group_size : 1024) * sizeof(float);
        hadamard_groups_float<<<grid, block, shmem, stream>>>(
            hadamard_input, (int)orgChunkCount, cfg.group_size, groups_per_chunk);
            
        kernelErr = cudaGetLastError();
        if (kernelErr != cudaSuccess) {
            if (d_adaptive_scales) cudaFree(d_adaptive_scales);
            if (allocated_temp) cudaFree(hadamard_input);
            return kernelErr;
        }
    }

    const void* compress_input_ptr = cfg.use_as_hadamard ? (const void*)hadamard_input : orgbuff;
    
    if (cfg.use_scale && cfg.group_size > 0) {
        if (orgDatatype == ncclFloat32 || (cfg.use_as_hadamard && orgDatatype == ncclBfloat16)) {
            int threads = 256;
            size_t shared_mem = threads * sizeof(float);
            for (int c = 0; c < numChunks; c++) {
                char* base = (char*)(*compbuff) + c * bytes_per_chunk;
                __nv_fp8_storage_t* d_fp8 = reinterpret_cast<__nv_fp8_storage_t*>(base);
                float* d_scales = reinterpret_cast<float*>(base + orgChunkCount);
                float* d_ada_scales = nullptr;

                if (cfg.use_as_hadamard) {
                    d_ada_scales = reinterpret_cast<float*>(base + orgChunkCount + groups_per_chunk * sizeof(float));
                    cudaMemcpyAsync(
                        d_ada_scales,
                        d_adaptive_scales + c * groups_per_chunk,
                        adaptive_scale_bytes_per_chunk,
                        cudaMemcpyDeviceToDevice, stream
                    );
                }

                const float* in_ptr = reinterpret_cast<const float*>(compress_input_ptr) + c * orgChunkCount;

                compute_scales_group_per_chunk<<<groups_per_chunk, threads, shared_mem, stream>>>(
                    in_ptr, d_scales, (int)orgChunkCount, groups_per_chunk, cfg.group_size, cfg.target_range);

                int gridSize = DIVUP((int)orgChunkCount, threads);
                compress_fp32_to_fp8_group_per_chunk<<<gridSize, threads, 0, stream>>>(
                    in_ptr, d_fp8, d_scales, (int)orgChunkCount, groups_per_chunk, cfg.group_size, saturate, fp8_type);
            }
        } else if (orgDatatype == ncclBfloat16 && !cfg.use_as_hadamard) {
            int threads = 256;
            size_t shared_mem = threads * sizeof(float);
            for (int c = 0; c < numChunks; c++) {
                char* base = (char*)(*compbuff) + c * bytes_per_chunk;
                __nv_fp8_storage_t* d_fp8 = reinterpret_cast<__nv_fp8_storage_t*>(base);
                float* d_scales = reinterpret_cast<float*>(base + orgChunkCount);

                compute_scales_group_per_chunk_bf16<<<groups_per_chunk, threads, shared_mem, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(orgbuff) + c * orgChunkCount,
                    d_scales, (int)orgChunkCount, groups_per_chunk, cfg.group_size, cfg.target_range);

                int gridSize = DIVUP((int)orgChunkCount, threads);
                compress_bf16_to_fp8_group_per_chunk<<<gridSize, threads, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(orgbuff) + c * orgChunkCount,
                    d_fp8, d_scales, (int)orgChunkCount, groups_per_chunk, cfg.group_size, saturate, fp8_type);
            }
        } else {
            goto simple_compression;
        }
    } else {
simple_compression:
        int blockSize = 256;
        int gridSize = DIVUP(orgChunkCount, blockSize);
        gridSize = std::max(gridSize, 1);
        for (int c = 0; c < numChunks; c++) {
            char* base = (char*)(*compbuff) + c * bytes_per_chunk;
            __nv_fp8_storage_t* d_fp8 = reinterpret_cast<__nv_fp8_storage_t*>(base);

            if (cfg.use_as_hadamard) {
                float* d_ada_scales = reinterpret_cast<float*>(base + orgChunkCount);
                cudaMemcpyAsync(
                    d_ada_scales,
                    d_adaptive_scales + c * groups_per_chunk,
                    adaptive_scale_bytes_per_chunk,
                    cudaMemcpyDeviceToDevice, stream
                );
            }

            const void* inptr_chunk = nullptr;
            if (cfg.use_as_hadamard) {
                inptr_chunk = reinterpret_cast<const float*>(hadamard_input) + c * orgChunkCount;
                compress_to_fp8_kernel<float><<<gridSize, blockSize, 0, stream>>>(
                    inptr_chunk, d_fp8, (int)orgChunkCount, saturate, fp8_type);
            } else {
                if (orgDatatype == ncclFloat32) {
                    inptr_chunk = reinterpret_cast<const float*>(orgbuff) + c * orgChunkCount;
                    compress_to_fp8_kernel<float><<<gridSize, blockSize, 0, stream>>>(inptr_chunk, d_fp8, (int)orgChunkCount, saturate, fp8_type);
                } else if (orgDatatype == ncclFloat16) {
                    inptr_chunk = reinterpret_cast<const __half*>(orgbuff) + c * orgChunkCount;
                    compress_to_fp8_kernel<__half><<<gridSize, blockSize, 0, stream>>>(inptr_chunk, d_fp8, (int)orgChunkCount, saturate, fp8_type);
                } else if (orgDatatype == ncclBfloat16) {
                    inptr_chunk = reinterpret_cast<const __nv_bfloat16*>(orgbuff) + c * orgChunkCount;
                    compress_to_fp8_kernel<__nv_bfloat16><<<gridSize, blockSize, 0, stream>>>(inptr_chunk, d_fp8, (int)orgChunkCount, saturate, fp8_type);
                } else {
                    kernelErr = cudaErrorNotSupported;
                    goto cleanup;
                }
            }
        }
    }

cleanup:
    if (allocated_temp && hadamard_input) cudaFree(hadamard_input);
    if (d_adaptive_scales) cudaFree(d_adaptive_scales);
    if (kernelErr != cudaSuccess) return kernelErr;
    return cudaStreamSynchronize(stream);
}

__hidden cudaError_t launchDecompress(void* decompbuff, const void* compbuff,
                                      size_t decompChunkCount, ncclDataType_t decompDatatype,
                                      size_t compChunkCount, ncclDataType_t compDatatype,
                                      size_t numChunks, void* config,
                                      cudaStream_t stream) {
    if (decompChunkCount == 0 || numChunks == 0) return cudaErrorInvalidValue;

    if (compDatatype == decompDatatype && compChunkCount == decompChunkCount) {
        size_t typeSize = getTypeSize(decompDatatype);
        size_t totalBytes = decompChunkCount * typeSize * numChunks;
        return cudaMemcpyAsync(decompbuff, compbuff, totalBytes, cudaMemcpyDeviceToDevice, stream);
    }

    fp8Config cfg = config ? *reinterpret_cast<fp8Config*>(config) : fp8Config();
    if (!config) {
        cfg.fp8_format = 0; cfg.saturation = 1; cfg.use_scale = 1;
        cfg.group_size = 128; cfg.target_range = 448.0f; cfg.safe_mode = 0;
        cfg.use_as_hadamard = 0;
        cfg.pivotSwap = 0; cfg.lambda = 1e-6f; cfg.fp8_max_val = 448.0f;
    }

    __nv_fp8_interpretation_t fp8_type = (cfg.fp8_format == 0) ? __NV_E4M3 : __NV_E5M2;
    int groups_per_chunk = 0;
    size_t adaptive_scale_bytes_per_chunk = 0;

    if (cfg.use_scale && cfg.group_size > 0) {
        groups_per_chunk = DIVUP((int)decompChunkCount, cfg.group_size);
    }
    if (cfg.use_as_hadamard) {
        if (groups_per_chunk == 0) groups_per_chunk = DIVUP((int)decompChunkCount, cfg.group_size);
        adaptive_scale_bytes_per_chunk = (size_t)groups_per_chunk * sizeof(float);
    }

    cudaError_t kernelErr = cudaSuccess;
    float* hadamard_out = nullptr;
    bool allocated_temp = false;

    if (cfg.use_scale && cfg.group_size > 0) {
        if (decompDatatype == ncclFloat32) {
            int blockSize = 256;
            for (int c = 0; c < numChunks; c++) {
                const char* base = (const char*)compbuff + c * compChunkCount;
                const __nv_fp8_storage_t* d_fp8 = reinterpret_cast<const __nv_fp8_storage_t*>(base);
                const float* d_scales = reinterpret_cast<const float*>(base + decompChunkCount);

                int gridSize = DIVUP((int)decompChunkCount, blockSize);
                decompress_fp8_to_fp32_group_per_chunk<<<gridSize, blockSize, 0, stream>>>(
                    d_fp8, reinterpret_cast<float*>(decompbuff) + c * decompChunkCount,
                    d_scales, (int)decompChunkCount, groups_per_chunk, cfg.group_size, fp8_type);
            }
        } else if (decompDatatype == ncclBfloat16) {
            int blockSize = 256;
            for (int c = 0; c < numChunks; c++) {
                const char* base = (const char*)compbuff + c * compChunkCount;
                const __nv_fp8_storage_t* d_fp8 = reinterpret_cast<const __nv_fp8_storage_t*>(base);
                const float* d_scales = reinterpret_cast<const float*>(base + decompChunkCount);

                int gridSize = DIVUP((int)decompChunkCount, blockSize);
                decompress_fp8_to_bf16_group_per_chunk<<<gridSize, blockSize, 0, stream>>>(
                    d_fp8, reinterpret_cast<__nv_bfloat16*>(decompbuff) + c * decompChunkCount,
                    d_scales, (int)decompChunkCount, groups_per_chunk, cfg.group_size, fp8_type);
            }
        } else {
            goto simple_decompression;
        }
    } else {
simple_decompression:
        int blockSize = 256;
        int gridSize = DIVUP(decompChunkCount, blockSize);
        gridSize = std::max(gridSize, 1);
        for (int c = 0; c < numChunks; c++) {
            const char* base = (const char*)compbuff + c * compChunkCount;
            const __nv_fp8_storage_t* d_fp8 = reinterpret_cast<const __nv_fp8_storage_t*>(base);

            if (decompDatatype == ncclFloat32) {
                decompress_from_fp8_kernel<float><<<gridSize, blockSize, 0, stream>>>(
                    d_fp8, reinterpret_cast<float*>(decompbuff) + c * decompChunkCount, (int)decompChunkCount, fp8_type);
            } else if (decompDatatype == ncclFloat16) {
                decompress_from_fp8_kernel<__half><<<gridSize, blockSize, 0, stream>>>(
                    d_fp8, reinterpret_cast<__half*>(decompbuff) + c * decompChunkCount, (int)decompChunkCount, fp8_type);
            } else if (decompDatatype == ncclBfloat16) {
                decompress_from_fp8_kernel<__nv_bfloat16><<<gridSize, blockSize, 0, stream>>>(
                    d_fp8, reinterpret_cast<__nv_bfloat16*>(decompbuff) + c * decompChunkCount, (int)decompChunkCount, fp8_type);
            } else {
                return cudaErrorNotSupported;
            }
        }
    }

    if (cfg.use_as_hadamard) {
        if (cfg.pivotSwap) return cudaErrorNotSupported;

        float* inv_scale_input = nullptr;
        if (decompDatatype == ncclFloat32) {
            inv_scale_input = reinterpret_cast<float*>(decompbuff);
        } else if (decompDatatype == ncclBfloat16) {
            size_t totalCount = decompChunkCount * numChunks;
            kernelErr = cudaMalloc(&hadamard_out, totalCount * sizeof(float));
            if (kernelErr != cudaSuccess) return kernelErr;
            allocated_temp = true;

            int block = 256;
            int grid = DIVUP((int)totalCount, block);
            bf16_to_float_kernel<<<grid, block, 0, stream>>>((const __nv_bfloat16*)decompbuff, hadamard_out, totalCount);
            
            inv_scale_input = hadamard_out;
        } else {
            return cudaErrorNotSupported;
        }

        int grid2 = (int)numChunks * groups_per_chunk;
        int block2 = (cfg.group_size <= 1024) ? cfg.group_size : 1024;
        block2 = std::max(block2, 32);
        size_t shmem = ((cfg.group_size <= 1024) ? cfg.group_size : 1024) * sizeof(float);
        hadamard_groups_float<<<grid2, block2, shmem, stream>>>(
            inv_scale_input, (int)decompChunkCount, cfg.group_size, groups_per_chunk);

        size_t groups_total = (size_t)numChunks * groups_per_chunk;
        float* d_adaptive_scales = nullptr;
        kernelErr = cudaMalloc(&d_adaptive_scales, groups_total * sizeof(float));
        if (kernelErr != cudaSuccess) goto decomp_cleanup;

        for (int c = 0; c < numChunks; c++) {
            const char* base = (const char*)compbuff + c * compChunkCount;
            const float* d_ada_scales = reinterpret_cast<const float*>(
                base + decompChunkCount + (cfg.use_scale ? groups_per_chunk * sizeof(float) : 0)
            );
            cudaMemcpyAsync(
                d_adaptive_scales + c * groups_per_chunk,
                d_ada_scales,
                adaptive_scale_bytes_per_chunk,
                cudaMemcpyDeviceToDevice, stream
            );
        }

        int inv_apply_threads = 256;
        int inv_apply_grid = DIVUP((int)(decompChunkCount * numChunks), inv_apply_threads);
        apply_adaptive_inv_scale<<<inv_apply_grid, inv_apply_threads, 0, stream>>>(
            inv_scale_input, d_adaptive_scales, (int)decompChunkCount, (int)numChunks, cfg.group_size);

        cudaFree(d_adaptive_scales);

        if (decompDatatype == ncclBfloat16) {
            size_t totalCount = decompChunkCount * numChunks;
            int block = 256;
            int grid = DIVUP((int)totalCount, block);
            float_to_bf16_kernel<<<grid, block, 0, stream>>>(hadamard_out, (__nv_bfloat16*)decompbuff, totalCount);
        }
    }

decomp_cleanup:
    if (allocated_temp && hadamard_out) cudaFree(hadamard_out);
    if (kernelErr != cudaSuccess) return kernelErr;

    return cudaStreamSynchronize(stream);
}

extern "C" const ncclCompressor_t taco {
    .name = "taco",
    .compress = launchCompress,
    .decompress = launchDecompress,
    .decompReduce = nullptr,
    .decompReduceComp = nullptr,
    .parseConfig = parseFp8Config
};