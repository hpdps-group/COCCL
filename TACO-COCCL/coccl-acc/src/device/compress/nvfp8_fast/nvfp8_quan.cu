#include "nvfp8_quan.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <nvtx3/nvToolsExt.h>
#include "nccl.h"

// =============================================================================
// Macros & Constants
// =============================================================================

#ifndef HAS_CUDA_MALLOC_ASYNC
#if CUDART_VERSION >= 11020
#define HAS_CUDA_MALLOC_ASYNC 1
#else
#define HAS_CUDA_MALLOC_ASYNC 0
#endif
#endif

#define WARP_SIZE 32
#define MAX_THREADS 1024
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#define __hidden __attribute__((visibility("hidden")))

// =============================================================================
// Host Helper Functions
// =============================================================================

inline cudaError_t safeCudaMallocAsync(void** ptr, size_t bytes, cudaStream_t stream) {
#if HAS_CUDA_MALLOC_ASYNC
    return cudaMallocAsync(ptr, bytes, stream);
#else
    return cudaMalloc(ptr, bytes);
#endif
}

inline cudaError_t safeCudaFreeAsync(void* ptr, cudaStream_t stream) {
#if HAS_CUDA_MALLOC_ASYNC
    return cudaFreeAsync(ptr, stream);
#else
    return cudaFree(ptr);
#endif
}

inline size_t getTypeSize(ncclDataType_t type) {
    switch (type) {
        case ncclFloat32:  return 4;
        case ncclFloat16:  return 2;
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
    int use_as_hadamard;
    float lambda;
    float fp8_max_val;
    int pivotSwap;
};

__hidden void parseFp8Config(const char* configFile, void** compConfig, int nodes, int devicesPerNodes) {
    *compConfig = (void*)malloc(sizeof(fp8Config));
    fp8Config* config = reinterpret_cast<fp8Config*>(*compConfig);

    // Default Configuration
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
    if (!configPairs) return;

    for (int i = 0; i < configPairCount; i++) {
        const char* key = configPairs[i].first;
        const char* val = configPairs[i].second;

        if (!strcmp(key, "fp8_type") || !strcmp(key, "fp8_format")) {
            config->fp8_format = (!strcmp(val, "E5M2")) ? 1 : 0;
            config->fp8_max_val = (config->fp8_format == 1) ? 57344.0f : 448.0f;
        } else if (!strcmp(key, "saturate") || !strcmp(key, "saturation")) {
            config->saturation = atoi(val);
        } else if (!strcmp(key, "group_size")) {
            config->group_size = atoi(val);
        } else if (!strcmp(key, "target_range")) {
            config->target_range = atof(val);
        } else if (!strcmp(key, "safe_mode")) {
            config->safe_mode = atoi(val);
        } else if (!strcmp(key, "as_hadamard")) {
            config->use_as_hadamard = atoi(val);
        } else if (!strcmp(key, "lambda")) {
            config->lambda = atof(val);
        } else if (!strcmp(key, "fp8_max_val")) {
            config->fp8_max_val = atof(val);
        }
    }
}

// =============================================================================
// Device Helper Functions (Math & Reductions)
// =============================================================================

__device__ __forceinline__ float safe_divide(float a, float b) {
    return (fabsf(b) < 1e-12f) ? 0.0f : (a / b);
}

__device__ __forceinline__ float clampf(float x, float min_val, float max_val) {
    return fmaxf(fminf(x, max_val), min_val);
}

__device__ __forceinline__ float check_nan_inf(float x) {
    return (isnan(x) || isinf(x)) ? 0.0f : x;
}

__device__ __forceinline__ float clamp_fp8(float x) {
    return fmaxf(fminf(x, 448.0f), -448.0f);
}

// Warp Reduction Sum
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Warp Reduction Max
__device__ __forceinline__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// =============================================================================
// Fused Kernels
// =============================================================================

// Super-Fused Kernel: Stats + Scale + Hadamard + Quantization
template<typename T, int GROUP_SIZE>
__global__ void __launch_bounds__(GROUP_SIZE) fused_ash_compress_kernel(
    const void* __restrict__ input,
    void* __restrict__ output,
    float* __restrict__ quant_scales,    // Not strictly needed if pointer math is done inside
    float* __restrict__ ada_scales_out,  // Not strictly needed if pointer math is done inside
    int orgChunkCount,
    size_t in_chunk_stride_bytes,
    size_t out_chunk_stride_bytes,
    size_t scale_chunk_stride_bytes,
    float target_range,
    float lambda,
    float fp8_max_val,
    __nv_saturation_t saturate,
    __nv_fp8_interpretation_t fp8_type)
{
    // Grid: x = group_id, y = chunk_id
    int group_id = blockIdx.x;
    int chunk_id = blockIdx.y;
    int tid = threadIdx.x;

    // Pointer Arithmetics
    const char* in_base = (const char*)input + chunk_id * in_chunk_stride_bytes;
    char* out_base = (char*)output + chunk_id * out_chunk_stride_bytes;

    // Scales pointers for this chunk (Layout: [Data] [QuantScales] [AdaScales])
    float* chunk_q_scales = (float*)(out_base + orgChunkCount);
    float* chunk_ada_scales = (float*)(out_base + orgChunkCount + scale_chunk_stride_bytes);

    int start_idx = group_id * GROUP_SIZE;
    int idx = start_idx + tid;

    // 1. Load Data
    float val = 0.0f;
    if (idx < orgChunkCount) {
        if (std::is_same<T, float>::value) {
            val = reinterpret_cast<const float*>(in_base)[idx];
        } else {
            val = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(in_base)[idx]);
        }
    }

    // 2. Compute Variance (Reductions)
    float val_sq = val * val;
    val_sq = warpReduceSum(val_sq); // Intra-warp

    static __shared__ float shared_stats[32];
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (lane == 0) shared_stats[warp_id] = val_sq;
    __syncthreads();

    // Block Reduction for Variance
    float group_sum_sq = (tid < (GROUP_SIZE / WARP_SIZE)) ? shared_stats[lane] : 0.0f;
    if (warp_id == 0) group_sum_sq = warpReduceSum(group_sum_sq);

    // Compute & Store Adaptive Scale
    __shared__ float ada_scale;
    if (tid == 0) {
        int count = min(GROUP_SIZE, orgChunkCount - start_idx);
        float var = (group_sum_sq / (float)count) + lambda;
        if (var < 1e-12f) var = 1e-12f;
        
        float s = fp8_max_val * rsqrtf(var);
        ada_scale = clampf(s, 1e-3f, 1e3f);

        // Store to global (for decoder)
        chunk_ada_scales[group_id] = ada_scale;
    }
    __syncthreads();

    // 3. Apply Adaptive Scale
    val *= ada_scale;

    // 4. Register-Level Hadamard (Steps 1, 2, 4, 8, 16)
    #pragma unroll
    for (int s = 1; s < 32; s *= 2) {
        float neighbor = __shfl_xor_sync(0xffffffff, val, s);
        if (tid & s) val = neighbor - val;
        else         val = val + neighbor;
    }

    // 5. Shared-Mem Hadamard (Steps 32+)
    if (GROUP_SIZE > 32) {
        extern __shared__ float smem_data[];
        smem_data[tid] = val;
        __syncthreads();

        for (int s = 32; s < GROUP_SIZE; s *= 2) {
            int neighbor_idx = tid ^ s;
            float neighbor = smem_data[neighbor_idx];

            float new_val;
            if (tid & s) new_val = neighbor - val;
            else         new_val = val + neighbor;

            __syncthreads();
            val = new_val;
            smem_data[tid] = val;
            __syncthreads();
        }
    }

    // Normalize
    val *= rsqrtf((float)GROUP_SIZE);

    // 6. Compute Quant Scale (Max Reduction)
    float abs_val = fabsf(val);
    abs_val = warpReduceMax(abs_val);

    if (lane == 0) shared_stats[warp_id] = abs_val;
    __syncthreads();

    float group_max = (tid < (GROUP_SIZE / WARP_SIZE)) ? shared_stats[lane] : 0.0f;
    if (warp_id == 0) group_max = warpReduceMax(group_max);

    __shared__ float quant_scale;
    if (tid == 0) {
        float mv = (group_max < 1e-12f) ? 1e-12f : group_max;
        float s = safe_divide(mv, target_range);
        quant_scale = clampf(s, 1e-12f, 1e6f);
        chunk_q_scales[group_id] = quant_scale;
    }
    __syncthreads();

    // 7. Quantize & Pack (Coalesced Write)
    val = check_nan_inf(val);
    float scaled = clamp_fp8(safe_divide(val, quant_scale));
    unsigned char q8 = (unsigned char)__nv_cvt_float_to_fp8(scaled, saturate, fp8_type);

    // Pack 4 bytes -> 1 int using shuffle to optimize global memory writes
    unsigned int packed = (unsigned int)q8;
    unsigned int b1 = __shfl_down_sync(0xffffffff, packed, 1);
    unsigned int b2 = __shfl_down_sync(0xffffffff, packed, 2);
    unsigned int b3 = __shfl_down_sync(0xffffffff, packed, 3);

    // Only one thread every 4 threads writes 32 bits
    if ((tid & 3) == 0 && (idx + 3 < orgChunkCount)) {
        unsigned int final_int = packed | (b1 << 8) | (b2 << 16) | (b3 << 24);
        reinterpret_cast<unsigned int*>(out_base)[idx / 4] = final_int;
    }
    // Tail handling
    else if (idx < orgChunkCount && (tid & 3) != 0 && (idx / 4 != (idx + 3) / 4)) {
        ((__nv_fp8_storage_t*)out_base)[idx] = (__nv_fp8_storage_t)q8;
    }
}

// Fused Decompress Kernel
template<typename T, int GROUP_SIZE>
__global__ void __launch_bounds__(GROUP_SIZE) fused_ash_decompress_kernel(
    const void* __restrict__ input,
    void* __restrict__ output,
    const float* __restrict__ q_scales_base,   // Not strictly needed
    const float* __restrict__ ada_scales_base, // Not strictly needed
    int orgChunkCount,
    size_t in_chunk_stride_bytes,
    size_t out_chunk_stride_bytes,
    size_t scale_chunk_stride_bytes,
    __nv_fp8_interpretation_t fp8_type)
{
    int group_id = blockIdx.x;
    int chunk_id = blockIdx.y;
    int tid = threadIdx.x;

    const char* in_base = (const char*)input + chunk_id * in_chunk_stride_bytes;
    char* out_base = (char*)output + chunk_id * out_chunk_stride_bytes;

    // Correct offsets for scales in the compressed buffer
    // Layout per chunk: [FP8 Data] [QuantScales] [AdaScales]
    const float* chunk_q_scales = (const float*)(in_base + orgChunkCount);
    const float* chunk_ada_scales = (const float*)(in_base + orgChunkCount + scale_chunk_stride_bytes);

    int idx = group_id * GROUP_SIZE + tid;

    // 1. Load FP8 & Dequant
    float val = 0.0f;
    if (idx < orgChunkCount) {
        __nv_fp8_storage_t raw = ((const __nv_fp8_storage_t*)in_base)[idx];
        __half_raw hr = __nv_cvt_fp8_to_halfraw(raw, fp8_type);
        float f = __half2float(*reinterpret_cast<__half*>(&hr));
        float qs = chunk_q_scales[group_id];
        val = f * qs;
    }

    // 2. Inverse Hadamard
    // Register-level (Steps 1-16)
    #pragma unroll
    for (int s = 1; s < 32; s *= 2) {
        float neighbor = __shfl_xor_sync(0xffffffff, val, s);
        if (tid & s) val = neighbor - val;
        else         val = val + neighbor;
    }

    // Shared-mem (Steps 32+)
    if (GROUP_SIZE > 32) {
        extern __shared__ float smem_data[];
        smem_data[tid] = val;
        __syncthreads();

        for (int s = 32; s < GROUP_SIZE; s *= 2) {
            int neighbor_idx = tid ^ s;
            float neighbor = smem_data[neighbor_idx];
            
            float new_val;
            if (tid & s) new_val = neighbor - val;
            else         new_val = val + neighbor;
            
            __syncthreads();
            val = new_val;
            smem_data[tid] = val;
            __syncthreads();
        }
    }
    val *= rsqrtf((float)GROUP_SIZE);

    // 3. Inverse Adaptive Scale & Store
    if (idx < orgChunkCount) {
        float ada_s = chunk_ada_scales[group_id];
        if (ada_s < 1e-12f) ada_s = 1e-12f;
        val = check_nan_inf(val) / ada_s;

        if (std::is_same<T, float>::value) {
            reinterpret_cast<float*>(out_base)[idx] = val;
        } else {
            reinterpret_cast<__nv_bfloat16*>(out_base)[idx] = __float2bfloat16(val);
        }
    }
}

// =============================================================================
// Launchers
// =============================================================================

// Helper for calling the template kernel based on datatype and group size
template<int GROUP_SIZE>
void launch_compress_template(
    ncclDataType_t datatype, const void* orgbuff, void* compbuff, 
    int orgChunkCount, size_t in_stride, size_t out_stride, size_t quant_scale_bytes,
    float target_range, float lambda, float fp8_max_val, 
    __nv_saturation_t saturate, __nv_fp8_interpretation_t fp8_type,
    dim3 grid, size_t smem, cudaStream_t stream) 
{
    if (datatype == ncclFloat32) {
        fused_ash_compress_kernel<float, GROUP_SIZE><<<grid, GROUP_SIZE, smem, stream>>>(
            orgbuff, compbuff, nullptr, nullptr, orgChunkCount,
            in_stride, out_stride, quant_scale_bytes,
            target_range, lambda, fp8_max_val, saturate, fp8_type);
    } else {
        fused_ash_compress_kernel<__nv_bfloat16, GROUP_SIZE><<<grid, GROUP_SIZE, smem, stream>>>(
            orgbuff, compbuff, nullptr, nullptr, orgChunkCount,
            in_stride, out_stride, quant_scale_bytes,
            target_range, lambda, fp8_max_val, saturate, fp8_type);
    }
}

__hidden cudaError_t launchCompress(const void* orgbuff, void** compbuff,
                                    size_t orgChunkCount, ncclDataType_t orgDatatype,
                                    size_t* compChunkCount, ncclDataType_t* compDatatype,
                                    size_t numChunks, void* config,
                                    cudaMemPool_t compMemPool, cudaStream_t stream) 
{
    if (orgChunkCount == 0 || numChunks == 0) return cudaSuccess;

    // Parse Config
    fp8Config cfg = config ? *reinterpret_cast<fp8Config*>(config) : fp8Config();
    if (!config) {
        cfg.group_size = 128; cfg.fp8_format = 0; cfg.saturation = 1;
        cfg.use_as_hadamard = 1; cfg.target_range = 448.0f; 
        cfg.lambda = 1e-6f; cfg.fp8_max_val = 448.0f;
    }

    *compDatatype = ncclUint8;
    __nv_fp8_interpretation_t fp8_type = (cfg.fp8_format == 0) ? __NV_E4M3 : __NV_E5M2;
    __nv_saturation_t saturate = (cfg.saturation == 1) ? __NV_SATFINITE : __NV_NOSAT;

    // Calculate Sizes
    int groups_per_chunk = DIVUP((int)orgChunkCount, cfg.group_size);
    size_t quant_scale_bytes = groups_per_chunk * sizeof(float);
    size_t ada_scale_bytes = groups_per_chunk * sizeof(float);
    size_t total_out_chunk_bytes = orgChunkCount + quant_scale_bytes + ada_scale_bytes;

    *compChunkCount = total_out_chunk_bytes;

    if (*compbuff == nullptr) {
        size_t totalBytes = total_out_chunk_bytes * numChunks;
        cudaError_t err = safeCudaMallocAsync((void**)compbuff, totalBytes, stream);
        if (err != cudaSuccess) return err;
    }

    // Launch Config
    dim3 grid(groups_per_chunk, (int)numChunks, 1);
    
    // Shared memory: Only needed if group_size > 32
    size_t smem = (cfg.group_size > 32) ? cfg.group_size * sizeof(float) : 0;
    
    size_t in_stride = orgChunkCount * getTypeSize(orgDatatype);
    size_t out_stride = total_out_chunk_bytes;

    // Dispatch to specific template instance based on group_size
    switch (cfg.group_size) {
        case 32:
            launch_compress_template<32>(
                orgDatatype, orgbuff, *compbuff, (int)orgChunkCount, 
                in_stride, out_stride, quant_scale_bytes, 
                cfg.target_range, cfg.lambda, cfg.fp8_max_val, saturate, fp8_type,
                grid, smem, stream);
            break;
        case 64:
            launch_compress_template<64>(
                orgDatatype, orgbuff, *compbuff, (int)orgChunkCount, 
                in_stride, out_stride, quant_scale_bytes, 
                cfg.target_range, cfg.lambda, cfg.fp8_max_val, saturate, fp8_type,
                grid, smem, stream);
            break;
        case 128:
            launch_compress_template<128>(
                orgDatatype, orgbuff, *compbuff, (int)orgChunkCount, 
                in_stride, out_stride, quant_scale_bytes, 
                cfg.target_range, cfg.lambda, cfg.fp8_max_val, saturate, fp8_type,
                grid, smem, stream);
            break;
        case 256:
            launch_compress_template<256>(
                orgDatatype, orgbuff, *compbuff, (int)orgChunkCount, 
                in_stride, out_stride, quant_scale_bytes, 
                cfg.target_range, cfg.lambda, cfg.fp8_max_val, saturate, fp8_type,
                grid, smem, stream);
            break;
        case 512:
            launch_compress_template<512>(
                orgDatatype, orgbuff, *compbuff, (int)orgChunkCount, 
                in_stride, out_stride, quant_scale_bytes, 
                cfg.target_range, cfg.lambda, cfg.fp8_max_val, saturate, fp8_type,
                grid, smem, stream);
            break;
        default:
            printf("[NVFP8 Error] Unsupported group_size: %d. Supported: 32, 64, 128, 256, 512.\n", cfg.group_size);
            return cudaErrorInvalidValue;
    }

    return cudaGetLastError();
}

template<int GROUP_SIZE>
void launch_decompress_template(
    ncclDataType_t datatype, void* decompbuff, const void* compbuff, 
    int decompChunkCount, size_t in_stride, size_t out_stride, size_t scale_chunk_stride_bytes,
    __nv_fp8_interpretation_t fp8_type,
    dim3 grid, size_t smem, cudaStream_t stream)
{
    if (datatype == ncclFloat32) {
        fused_ash_decompress_kernel<float, GROUP_SIZE><<<grid, GROUP_SIZE, smem, stream>>>(
            compbuff, decompbuff, nullptr, nullptr, decompChunkCount,
            in_stride, out_stride, scale_chunk_stride_bytes, fp8_type);
    } else {
        fused_ash_decompress_kernel<__nv_bfloat16, GROUP_SIZE><<<grid, GROUP_SIZE, smem, stream>>>(
            compbuff, decompbuff, nullptr, nullptr, decompChunkCount,
            in_stride, out_stride, scale_chunk_stride_bytes, fp8_type);
    }
}

__hidden cudaError_t launchDecompress(void* decompbuff, const void* compbuff,
                                      size_t decompChunkCount, ncclDataType_t decompDatatype,
                                      size_t compChunkCount, ncclDataType_t compDatatype,
                                      size_t numChunks, void* config,
                                      cudaStream_t stream) 
{
    if (decompChunkCount == 0 || numChunks == 0) return cudaSuccess;

    fp8Config cfg = config ? *reinterpret_cast<fp8Config*>(config) : fp8Config();
    if (!config) { cfg.group_size = 128; cfg.fp8_format = 0; }

    __nv_fp8_interpretation_t fp8_type = (cfg.fp8_format == 0) ? __NV_E4M3 : __NV_E5M2;
    int groups_per_chunk = DIVUP((int)decompChunkCount, cfg.group_size);
    size_t quant_scale_bytes = groups_per_chunk * sizeof(float);

    dim3 grid(groups_per_chunk, (int)numChunks, 1);
    size_t smem = (cfg.group_size > 32) ? cfg.group_size * sizeof(float) : 0;

    size_t in_stride = compChunkCount; // passed from compress output size
    size_t out_stride = decompChunkCount * getTypeSize(decompDatatype);

    // Dispatch to specific template instance based on group_size
    switch (cfg.group_size) {
        case 32:
            launch_decompress_template<32>(
                decompDatatype, decompbuff, compbuff, (int)decompChunkCount,
                in_stride, out_stride, quant_scale_bytes, fp8_type,
                grid, smem, stream);
            break;
        case 64:
            launch_decompress_template<64>(
                decompDatatype, decompbuff, compbuff, (int)decompChunkCount,
                in_stride, out_stride, quant_scale_bytes, fp8_type,
                grid, smem, stream);
            break;
        case 128:
            launch_decompress_template<128>(
                decompDatatype, decompbuff, compbuff, (int)decompChunkCount,
                in_stride, out_stride, quant_scale_bytes, fp8_type,
                grid, smem, stream);
            break;
        case 256:
            launch_decompress_template<256>(
                decompDatatype, decompbuff, compbuff, (int)decompChunkCount,
                in_stride, out_stride, quant_scale_bytes, fp8_type,
                grid, smem, stream);
            break;
        case 512:
            launch_decompress_template<512>(
                decompDatatype, decompbuff, compbuff, (int)decompChunkCount,
                in_stride, out_stride, quant_scale_bytes, fp8_type,
                grid, smem, stream);
            break;
        default:
            printf("[NVFP8 Error] Unsupported group_size: %d. Supported: 32, 64, 128, 256, 512.\n", cfg.group_size);
            return cudaErrorInvalidValue;
    }

    return cudaGetLastError();
}

extern "C" const ncclCompressor_t nvfp8 {
    .name = "nvfp8",
    .compress = launchCompress,
    .decompress = launchDecompress,
    .decompReduce = nullptr,
    .decompReduceComp = nullptr,
    .parseConfig = parseFp8Config
};