#include "compressor_plugin/coccl_compressor_plugin.h"

#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <type_traits>

#define WARP_SIZE 32
#define DIVUP(x, y) (((x) + (y) - 1) / (y))

namespace {

struct TacoConfig {
    int fp8Format = 0;
    bool saturate = true;
    int groupSize = 128;
    float targetRange = 448.0f;
    float lambda = 1e-6f;
    float fp8MaxValue = 0.0f;

    float maxValue() const {
        return fp8MaxValue > 0.0f ? fp8MaxValue
                                  : (fp8Format == 0 ? 448.0f : 57344.0f);
    }
};

bool validGroupSize(int value) {
    return value == 32 || value == 64 || value == 128 || value == 256 ||
           value == 512;
}

template <typename Shape>
bool compressedBytes(const Shape& input, const TacoConfig& config,
                     size_t* bytes) {
    if (bytes == nullptr || input.elementsPerChunk() == 0 ||
        !validGroupSize(config.groupSize)) {
        return false;
    }
    const size_t groups =
        DIVUP(input.elementsPerChunk(), (size_t)config.groupSize);
    size_t scaleBytes = 0;
    size_t metadataBytes = 0;
    size_t outputChunkBytes = 0;
    return coccl::checkedMultiply(groups, sizeof(float), &scaleBytes) &&
        coccl::checkedMultiply(scaleBytes, 2, &metadataBytes) &&
        coccl::checkedAdd(input.elementsPerChunk(), metadataBytes,
                          &outputChunkBytes) &&
        coccl::checkedMultiply(outputChunkBytes, input.chunks(), bytes);
}


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


// Super-Fused Kernel: Stats + Scale + Hadamard + Quantization
template<typename T, int GROUP_SIZE>
__global__ void __launch_bounds__(GROUP_SIZE) fused_ash_compress_kernel(
    const void* __restrict__ input,
    void* __restrict__ output,
    float* __restrict__ quant_scales,    // Not strictly needed
    float* __restrict__ ada_scales_out,  // Not strictly needed
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
    int group_id = blockIdx.x;
    int chunk_id = blockIdx.y;
    int tid = threadIdx.x;

    const char* in_base = (const char*)input + chunk_id * in_chunk_stride_bytes;
    char* out_base = (char*)output + chunk_id * out_chunk_stride_bytes;

    float* chunk_q_scales = (float*)(out_base + orgChunkCount);
    float* chunk_ada_scales = (float*)(out_base + orgChunkCount + scale_chunk_stride_bytes);

    int start_idx = group_id * GROUP_SIZE;
    int idx = start_idx + tid;

    float val = 0.0f;
    if (idx < orgChunkCount) {
        if (std::is_same<T, float>::value) {
            val = reinterpret_cast<const float*>(in_base)[idx];
        } else {
            val = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(in_base)[idx]);
        }
    }

    float val_sq = val * val;
    val_sq = warpReduceSum(val_sq); 

    static __shared__ float shared_stats[32];
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (lane == 0) shared_stats[warp_id] = val_sq;
    __syncthreads();

    float group_sum_sq = (tid < (GROUP_SIZE / WARP_SIZE)) ? shared_stats[lane] : 0.0f;
    if (warp_id == 0) group_sum_sq = warpReduceSum(group_sum_sq);

    __shared__ float ada_scale;
    if (tid == 0) {
        int count = min(GROUP_SIZE, orgChunkCount - start_idx);
        float var = (group_sum_sq / (float)count) + lambda;
        if (var < 1e-12f) var = 1e-12f;
        
        float s = fp8_max_val * rsqrtf(var);
        ada_scale = clampf(s, 1e-3f, 1e3f);

        chunk_ada_scales[group_id] = ada_scale;
    }
    __syncthreads();

    val *= ada_scale;

    #pragma unroll
    for (int s = 1; s < 32; s *= 2) {
        float neighbor = __shfl_xor_sync(0xffffffff, val, s);
        if (tid & s) val = neighbor - val;
        else         val = val + neighbor;
    }

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

    val = check_nan_inf(val);
    float scaled = clamp_fp8(safe_divide(val, quant_scale));
    unsigned char q8 = (unsigned char)__nv_cvt_float_to_fp8(scaled, saturate, fp8_type);

    unsigned int packed = (unsigned int)q8;
    unsigned int b1 = __shfl_down_sync(0xffffffff, packed, 1);
    unsigned int b2 = __shfl_down_sync(0xffffffff, packed, 2);
    unsigned int b3 = __shfl_down_sync(0xffffffff, packed, 3);

    const int wordStart = idx - (tid & 3);
    const bool fullWord = wordStart + 3 < orgChunkCount;
    if ((tid & 3) == 0 && fullWord) {
        unsigned int final_int = packed | (b1 << 8) | (b2 << 16) | (b3 << 24);
        reinterpret_cast<unsigned int*>(out_base)[idx / 4] = final_int;
    } else if (idx < orgChunkCount && !fullWord) {
        ((__nv_fp8_storage_t*)out_base)[idx] = (__nv_fp8_storage_t)q8;
    }
}

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

    float val = 0.0f;
    if (idx < orgChunkCount) {
        __nv_fp8_storage_t raw = ((const __nv_fp8_storage_t*)in_base)[idx];
        __half_raw hr = __nv_cvt_fp8_to_halfraw(raw, fp8_type);
        float f = __half2float(*reinterpret_cast<__half*>(&hr));
        float qs = chunk_q_scales[group_id];
        val = f * qs;
    }

    #pragma unroll
    for (int s = 1; s < 32; s *= 2) {
        float neighbor = __shfl_xor_sync(0xffffffff, val, s);
        if (tid & s) val = neighbor - val;
        else         val = val + neighbor;
    }

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


template<int GROUP_SIZE>
void launchCompressKernel(
    ncclDataType_t datatype, const void* input, void* output,
    int elementsPerChunk, size_t inputStride, size_t outputStride,
    size_t scaleBytes, const TacoConfig& config, dim3 grid,
    size_t sharedBytes, cudaStream_t stream) {
    const __nv_fp8_interpretation_t fp8Type =
        config.fp8Format == 0 ? __NV_E4M3 : __NV_E5M2;
    const __nv_saturation_t saturation =
        config.saturate ? __NV_SATFINITE : __NV_NOSAT;
    if (datatype == ncclFloat32) {
        fused_ash_compress_kernel<float, GROUP_SIZE>
            <<<grid, GROUP_SIZE, sharedBytes, stream>>>(
                input, output, nullptr, nullptr, elementsPerChunk,
                inputStride, outputStride, scaleBytes, config.targetRange,
                config.lambda, config.maxValue(), saturation, fp8Type);
    } else {
        fused_ash_compress_kernel<__nv_bfloat16, GROUP_SIZE>
            <<<grid, GROUP_SIZE, sharedBytes, stream>>>(
                input, output, nullptr, nullptr, elementsPerChunk,
                inputStride, outputStride, scaleBytes, config.targetRange,
                config.lambda, config.maxValue(), saturation, fp8Type);
    }
}

template<int GROUP_SIZE>
void launchDecompressKernel(
    ncclDataType_t datatype, void* output, const void* input,
    int elementsPerChunk, size_t inputStride, size_t outputStride,
    size_t scaleBytes, const TacoConfig& config, dim3 grid,
    size_t sharedBytes, cudaStream_t stream) {
    const __nv_fp8_interpretation_t fp8Type =
        config.fp8Format == 0 ? __NV_E4M3 : __NV_E5M2;
    if (datatype == ncclFloat32) {
        fused_ash_decompress_kernel<float, GROUP_SIZE>
            <<<grid, GROUP_SIZE, sharedBytes, stream>>>(
                input, output, nullptr, nullptr, elementsPerChunk,
                inputStride, outputStride, scaleBytes, fp8Type);
    } else {
        fused_ash_decompress_kernel<__nv_bfloat16, GROUP_SIZE>
            <<<grid, GROUP_SIZE, sharedBytes, stream>>>(
                input, output, nullptr, nullptr, elementsPerChunk,
                inputStride, outputStride, scaleBytes, fp8Type);
    }
}

struct TacoCompressor {
    using Config = TacoConfig;

    static coccl::Status configure(coccl::ConfigReader& reader,
                                   Config& config,
                                   const coccl::ConfigContext&) {
        coccl::Status result =
            reader.getEnum("fp8Format", config.fp8Format,
                           {{"E4M3", 0}, {"E5M2", 1}})
                .get("saturate", config.saturate)
                .get("groupSize", config.groupSize, 32, 512)
                .get("targetRange", config.targetRange, FLT_MIN, FLT_MAX)
                .get("lambda", config.lambda, 0.0f, FLT_MAX)
                .get("fp8MaxValue", config.fp8MaxValue, 0.0f, FLT_MAX)
                .finish();
        if (result != ncclSuccess) return result;
        return validGroupSize(config.groupSize) ? ncclSuccess
                                                : ncclInvalidArgument;
    }

    static coccl::Status encodedSizeBound(
        const coccl::Shape& input, size_t* encodedBytes,
        const coccl::SizeContext& context) {
        if (context.operation() != cocclCompressorOperationCompress) {
            return ncclInvalidUsage;
        }
        return compressedBytes(input, context.config<Config>(), encodedBytes)
            ? ncclSuccess : ncclInvalidArgument;
    }

    static coccl::Status compress(const coccl::Input& input,
                                  coccl::Output& output,
                                  coccl::Context& context) {
        if ((input.datatype() != ncclFloat32 &&
             input.datatype() != ncclBfloat16) ||
            input.elementsPerChunk() == 0 ||
            input.elementsPerChunk() > INT_MAX || input.chunks() > INT_MAX) {
            return ncclInvalidArgument;
        }

        const Config& config = context.config<Config>();
        const size_t groups =
            DIVUP(input.elementsPerChunk(), (size_t)config.groupSize);
        size_t scaleBytes = 0;
        size_t metadataBytes = 0;
        size_t outputChunkBytes = 0;
        size_t outputBytes = 0;
        if (!compressedBytes(input, config, &outputBytes) ||
            !coccl::checkedMultiply(groups, sizeof(float), &scaleBytes) ||
            !coccl::checkedMultiply(scaleBytes, 2, &metadataBytes) ||
            !coccl::checkedAdd(input.elementsPerChunk(), metadataBytes,
                               &outputChunkBytes)) {
            return ncclInvalidArgument;
        }
        if (coccl::shouldPassthrough(input, outputBytes)) {
            return output.passthrough(input, context.stream());
        }
        if (outputBytes > output.capacityBytes()) return ncclInvalidArgument;

        const dim3 grid((unsigned)groups, (unsigned)input.chunks(), 1);
        const size_t sharedBytes = config.groupSize > 32
            ? (size_t)config.groupSize * sizeof(float)
            : 0;
        const size_t inputStride = input.bytes() / input.chunks();
        switch (config.groupSize) {
            case 32:
                launchCompressKernel<32>(
                    input.datatype(), input.data(), output.data(),
                    (int)input.elementsPerChunk(), inputStride,
                    outputChunkBytes, scaleBytes, config, grid, sharedBytes,
                    context.stream());
                break;
            case 64:
                launchCompressKernel<64>(
                    input.datatype(), input.data(), output.data(),
                    (int)input.elementsPerChunk(), inputStride,
                    outputChunkBytes, scaleBytes, config, grid, sharedBytes,
                    context.stream());
                break;
            case 128:
                launchCompressKernel<128>(
                    input.datatype(), input.data(), output.data(),
                    (int)input.elementsPerChunk(), inputStride,
                    outputChunkBytes, scaleBytes, config, grid, sharedBytes,
                    context.stream());
                break;
            case 256:
                launchCompressKernel<256>(
                    input.datatype(), input.data(), output.data(),
                    (int)input.elementsPerChunk(), inputStride,
                    outputChunkBytes, scaleBytes, config, grid, sharedBytes,
                    context.stream());
                break;
            case 512:
                launchCompressKernel<512>(
                    input.datatype(), input.data(), output.data(),
                    (int)input.elementsPerChunk(), inputStride,
                    outputChunkBytes, scaleBytes, config, grid, sharedBytes,
                    context.stream());
                break;
            default: return ncclInvalidArgument;
        }

        cudaError_t cudaResult = cudaGetLastError();
        if (cudaResult != cudaSuccess) return coccl::fromCuda(cudaResult);
        return output.commitBytes(outputBytes, input.chunks());
    }

    static coccl::Status decompress(const coccl::Input& input,
                                    coccl::Output& output,
                                    coccl::Context& context) {
        if ((output.datatype() != ncclFloat32 &&
             output.datatype() != ncclBfloat16) ||
            output.elements() == 0 || output.chunks() == 0 ||
            output.elements() / output.chunks() > INT_MAX ||
            input.chunks() != output.chunks() || input.chunks() > INT_MAX) {
            return ncclInvalidArgument;
        }

        const Config& config = context.config<Config>();
        const size_t elementsPerChunk = output.elements() / output.chunks();
        const size_t groups = DIVUP(elementsPerChunk,
                                    (size_t)config.groupSize);
        size_t scaleBytes = 0;
        size_t metadataBytes = 0;
        size_t inputChunkBytes = 0;
        size_t expectedInputBytes = 0;
        size_t outputStride = 0;
        if (!coccl::checkedMultiply(groups, sizeof(float), &scaleBytes) ||
            !coccl::checkedMultiply(scaleBytes, 2, &metadataBytes) ||
            !coccl::checkedAdd(elementsPerChunk, metadataBytes,
                               &inputChunkBytes) ||
            !coccl::checkedMultiply(inputChunkBytes, input.chunks(),
                                    &expectedInputBytes) ||
            expectedInputBytes != input.bytes() ||
            !coccl::checkedMultiply(elementsPerChunk,
                                    coccl::dataTypeSize(output.datatype()),
                                    &outputStride)) {
            return ncclInvalidArgument;
        }

        const dim3 grid((unsigned)groups, (unsigned)input.chunks(), 1);
        const size_t sharedBytes = config.groupSize > 32
            ? (size_t)config.groupSize * sizeof(float)
            : 0;
        switch (config.groupSize) {
            case 32:
                launchDecompressKernel<32>(
                    output.datatype(), output.data(), input.data(),
                    (int)elementsPerChunk, inputChunkBytes, outputStride,
                    scaleBytes, config, grid, sharedBytes, context.stream());
                break;
            case 64:
                launchDecompressKernel<64>(
                    output.datatype(), output.data(), input.data(),
                    (int)elementsPerChunk, inputChunkBytes, outputStride,
                    scaleBytes, config, grid, sharedBytes, context.stream());
                break;
            case 128:
                launchDecompressKernel<128>(
                    output.datatype(), output.data(), input.data(),
                    (int)elementsPerChunk, inputChunkBytes, outputStride,
                    scaleBytes, config, grid, sharedBytes, context.stream());
                break;
            case 256:
                launchDecompressKernel<256>(
                    output.datatype(), output.data(), input.data(),
                    (int)elementsPerChunk, inputChunkBytes, outputStride,
                    scaleBytes, config, grid, sharedBytes, context.stream());
                break;
            case 512:
                launchDecompressKernel<512>(
                    output.datatype(), output.data(), input.data(),
                    (int)elementsPerChunk, inputChunkBytes, outputStride,
                    scaleBytes, config, grid, sharedBytes, context.stream());
                break;
            default: return ncclInvalidArgument;
        }
        return coccl::fromCuda(cudaGetLastError());
    }
};

}  // namespace

COCCL_REGISTER_COMPRESSOR("taco", TacoCompressor);
