#include "compressor_plugin/coccl_compressor_plugin.h"

#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

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

// scaled is finite and clamped to [-448, 448] before this conversion.
__device__ __forceinline__ unsigned char tacoEncodeFp8(
    float scaled, __nv_saturation_t saturate,
    __nv_fp8_interpretation_t format) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    return __nv_cvt_float_to_fp8(scaled, saturate, format);
#else
    const unsigned int bits = __float_as_uint(scaled);
    const unsigned int magnitude = bits & 0x7fffffffU;
    const unsigned int sign = (bits >> 24) & 0x80U;
    unsigned int result;
    if (format == __NV_E4M3) {
        if (magnitude < 0x3c800000U) {
            result = __float2uint_rn(__uint_as_float(magnitude) * 512.0f);
        } else {
            result = ((magnitude + 0x7ffffU + ((magnitude >> 20) & 1U)) >> 20) - 960U;
        }
    } else {
        if (magnitude < 0x38800000U) {
            result = __float2uint_rn(__uint_as_float(magnitude) * 65536.0f);
        } else {
            result = ((magnitude + 0xfffffU + ((magnitude >> 21) & 1U)) >> 21) - 448U;
        }
    }
    return static_cast<unsigned char>(sign | result);
#endif
}

__device__ __forceinline__ float tacoDecodeFp8(
    unsigned char encoded, __nv_fp8_interpretation_t format) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    const __half_raw raw = __nv_cvt_fp8_to_halfraw(encoded, format);
    return __half2float(*reinterpret_cast<const __half*>(&raw));
#else
    const unsigned int magnitude = encoded & 0x7fU;
    const unsigned int sign = (static_cast<unsigned int>(encoded) & 0x80U) << 24;
    if (format == __NV_E5M2) {
        __half_raw raw;
        raw.x = magnitude > 0x7cU ? 0x7fffU : (static_cast<unsigned short>(encoded) << 8);
        return __half2float(*reinterpret_cast<const __half*>(&raw));
    }
    if (magnitude == 0x7fU) {
        __half_raw raw;
        raw.x = 0x7fffU;
        return __half2float(*reinterpret_cast<const __half*>(&raw));
    }
    if (magnitude < 8U) {
        const float value = float(magnitude) * (1.0f / 512.0f);
        return __uint_as_float(__float_as_uint(value) | sign);
    }
    return __uint_as_float(sign | ((magnitude + 960U) << 20));
#endif
}

template<int Bytes> struct TacoVector;
template<> struct TacoVector<16> { using Type = uint4; };
template<> struct TacoVector<8> { using Type = uint2; };
template<> struct TacoVector<4> { using Type = unsigned int; };
template<> struct TacoVector<2> { using Type = unsigned short; };
template<> struct TacoVector<1> { using Type = unsigned char; };

template<typename T, int Slots, bool FullAligned>
__device__ __forceinline__ void tacoLoad(const T* source, T (&values)[Slots], int count) {
    constexpr int bytes = sizeof(T) * Slots < 16 ? sizeof(T) * Slots : 16;
    using Vector = typename TacoVector<bytes>::Type;
    if (FullAligned || (count == Slots && (reinterpret_cast<uintptr_t>(source) & (bytes - 1)) == 0)) {
        #pragma unroll
        for (int i = 0; i < sizeof(T) * Slots / bytes; ++i)
            reinterpret_cast<Vector*>(values)[i] = reinterpret_cast<const Vector*>(source)[i];
    } else {
        #pragma unroll
        for (int i = 0; i < Slots; ++i) values[i] = i < count ? source[i] : T(0);
    }
}

template<typename T, int Slots, bool FullAligned>
__device__ __forceinline__ void tacoStore(T* output, const T (&values)[Slots], int count) {
    constexpr int bytes = sizeof(T) * Slots < 16 ? sizeof(T) * Slots : 16;
    using Vector = typename TacoVector<bytes>::Type;
    if (FullAligned || (count == Slots && (reinterpret_cast<uintptr_t>(output) & (bytes - 1)) == 0)) {
        #pragma unroll
        for (int i = 0; i < sizeof(T) * Slots / bytes; ++i)
            reinterpret_cast<Vector*>(output)[i] = reinterpret_cast<const Vector*>(values)[i];
    } else {
        #pragma unroll
        for (int i = 0; i < Slots; ++i) if (i < count) output[i] = values[i];
    }
}

// Each lane owns Slots consecutive elements. Keep the original butterfly stages.
template<int Width, int Slots>
__device__ __forceinline__ void tacoWarpHadamard(float (&values)[Slots]) {
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int span = 1; span < Width; span *= 2) {
        #pragma unroll
        for (int i = 0; i < Slots; ++i) if ((i & span) == 0) {
            const float a = values[i], b = values[i + span];
            values[i] = __fadd_rn(a, b);
            values[i + span] = __fsub_rn(a, b);
        }
    }
    #pragma unroll
    for (int step = 1; step < 32; step *= 2) {
        #pragma unroll
        for (int i = 0; i < Slots; ++i) {
            const float neighbor = __shfl_xor_sync(0xffffffff, values[i], step);
            values[i] = (lane & step) ? neighbor - values[i] : values[i] + neighbor;
        }
    }
    #pragma unroll
    for (int span = Width; span < Slots; span *= 2) {
        #pragma unroll
        for (int i = 0; i < Slots; ++i) if ((i & span) == 0) {
            const float a = values[i], b = values[i + span];
            values[i] = __fadd_rn(a, b);
            values[i + span] = __fsub_rn(a, b);
        }
    }
}

template<int Slots, bool Max>
__device__ __forceinline__ float tacoContiguousReduce(float (&values)[Slots]) {
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int offset = 16; offset >= Slots; offset /= 2) {
        #pragma unroll
        for (int i = 0; i < Slots; ++i) {
            const float other = __shfl_down_sync(0xffffffff, values[i], offset / Slots, 32 / Slots);
            if (Max) values[i] = fmaxf(values[i], other);
            else values[i] += other;
        }
    }
    #pragma unroll
    for (int offset = Slots / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0; i < offset; ++i) {
            if (Max) values[i] = fmaxf(values[i], values[i + offset]);
            else values[i] += values[i + offset];
        }
    }
    float partial = __shfl_sync(0xffffffff, values[0], (lane % Slots) * (32 / Slots));
    partial = lane < Slots ? partial : 0.0f;
    #pragma unroll
    for (int offset = 16; offset >= Slots; offset /= 2) {
        // These source lanes contain zero in the original second-level tree.
        if (Max) partial = fmaxf(partial, 0.0f);
        else partial += 0.0f;
    }
    #pragma unroll
    for (int offset = Slots / 2; offset > 0; offset /= 2) {
        const float other = __shfl_down_sync(0xffffffff, partial, offset);
        if (Max) partial = fmaxf(partial, other);
        else partial += other;
    }
    return partial;
}

template<typename T, int GROUP_SIZE, int BLOCK_THREADS, bool FullAligned>
__global__ void __launch_bounds__(BLOCK_THREADS) fused_ash_compress_kernel(
    const void* __restrict__ input, void* __restrict__ output,
    int orgChunkCount, size_t in_chunk_stride_bytes,
    size_t out_chunk_stride_bytes, size_t scale_chunk_stride_bytes,
    float target_range, float lambda, float fp8_max_val,
    __nv_saturation_t saturate, __nv_fp8_interpretation_t fp8_type) {
    constexpr int slots = GROUP_SIZE / 32;
    const int lane = threadIdx.x & 31;
    const int group = blockIdx.x * (BLOCK_THREADS / 32) + threadIdx.x / 32;
    const size_t start = size_t(group) * GROUP_SIZE;
    if (start >= size_t(orgChunkCount)) return;
    const char* in_base = static_cast<const char*>(input) + size_t(blockIdx.y) * in_chunk_stride_bytes;
    char* out_base = static_cast<char*>(output) + size_t(blockIdx.y) * out_chunk_stride_bytes;
    float* q_scales = reinterpret_cast<float*>(out_base + orgChunkCount);
    float* ada_scales = reinterpret_cast<float*>(out_base + orgChunkCount + scale_chunk_stride_bytes);
    const size_t first = start + lane * slots;
    const int available = FullAligned ? slots
        : (first < size_t(orgChunkCount) ? min(slots, orgChunkCount - int(first)) : 0);
    __align__(16) T input_values[slots];
    tacoLoad<T, slots, FullAligned>(reinterpret_cast<const T*>(in_base) + first, input_values, available);
    float values[slots];
    // Reproduce the original two-level reduction tree, including zero lanes.
    float squares[slots];
    #pragma unroll
    for (int i = 0; i < slots; ++i) {
        const float value = float(input_values[i]);
        values[i] = value;
        squares[i] = value * value;
    }
    const float group_sum = tacoContiguousReduce<slots, false>(squares);
    float ada_scale = 0.0f;
    if (lane == 0) {
        const int count = min(GROUP_SIZE, orgChunkCount - int(start));
        // Full groups divide by an exact power of two; keep the multiply rounded before adding lambda.
        const float mean = FullAligned
            ? __fmul_rn(group_sum, 1.0f / float(GROUP_SIZE))
            : group_sum / float(count);
        float var = mean + lambda;
        if (var < 1e-12f) var = 1e-12f;
        ada_scale = clampf(fp8_max_val * rsqrtf(var), 1e-3f, 1e3f);
        ada_scales[group] = ada_scale;
    }
    ada_scale = __shfl_sync(0xffffffff, ada_scale, 0);
    #pragma unroll
    for (int i = 0; i < slots; ++i) values[i] *= ada_scale;
    tacoWarpHadamard<slots>(values);
    float magnitudes[slots];
    #pragma unroll
    for (int i = 0; i < slots; ++i) {
        values[i] *= rsqrtf(float(GROUP_SIZE));
        magnitudes[i] = fabsf(values[i]);
    }
    const float group_max = tacoContiguousReduce<slots, true>(magnitudes);
    float quant_scale = 0.0f;
    if (lane == 0) {
        const float mv = group_max < 1e-12f ? 1e-12f : group_max;
        quant_scale = clampf(safe_divide(mv, target_range), 1e-12f, 1e6f);
        q_scales[group] = quant_scale;
    }
    quant_scale = __shfl_sync(0xffffffff, quant_scale, 0);
    __align__(16) unsigned char encoded_values[slots];
    #pragma unroll
    for (int i = 0; i < slots; ++i) {
        // quant_scale is already finite and clamped to [1e-12, 1e6].
        const float scaled = clamp_fp8(check_nan_inf(values[i]) / quant_scale);
        const unsigned char encoded = tacoEncodeFp8(scaled, saturate, fp8_type);
        encoded_values[i] = encoded;
    }
    tacoStore<unsigned char, slots, FullAligned>(reinterpret_cast<unsigned char*>(out_base) + first, encoded_values, available);
}

template<typename T, int GROUP_SIZE, int BLOCK_THREADS, bool FullAligned>
__global__ void __launch_bounds__(BLOCK_THREADS) fused_ash_decompress_kernel(
    const void* __restrict__ input, void* __restrict__ output,
    int orgChunkCount, size_t in_chunk_stride_bytes,
    size_t out_chunk_stride_bytes, size_t scale_chunk_stride_bytes,
    __nv_fp8_interpretation_t fp8_type) {
    constexpr int slots = GROUP_SIZE / 32;
    // Cap each lane's raw output tile at 16 bytes to keep vector stores coalesced.
    constexpr int width = slots * sizeof(T) < 16 ? slots : 16 / sizeof(T);
    constexpr int tiles = slots / width;
    const int lane = threadIdx.x & 31;
    const int group = blockIdx.x * (BLOCK_THREADS / 32) + threadIdx.x / 32;
    const size_t start = size_t(group) * GROUP_SIZE;
    if (start >= size_t(orgChunkCount)) return;
    const char* in_base = static_cast<const char*>(input) + size_t(blockIdx.y) * in_chunk_stride_bytes;
    char* out_base = static_cast<char*>(output) + size_t(blockIdx.y) * out_chunk_stride_bytes;
    const float* q_scales = reinterpret_cast<const float*>(in_base + orgChunkCount);
    const float* ada_scales = reinterpret_cast<const float*>(in_base + orgChunkCount + scale_chunk_stride_bytes);
    const float quant_scale = q_scales[group];
    __align__(16) unsigned char encoded_values[tiles][width];
    #pragma unroll
    for (int tile = 0; tile < tiles; ++tile) {
        const size_t first = start + tile * 32 * width + lane * width;
        const int available = FullAligned ? width
            : (first < size_t(orgChunkCount) ? min(width, orgChunkCount - int(first)) : 0);
        tacoLoad<unsigned char, width, FullAligned>(
            reinterpret_cast<const unsigned char*>(in_base) + first, encoded_values[tile], available);
    }
    float values[slots];
    #pragma unroll
    for (int i = 0; i < slots; ++i) {
        const size_t index = start + (i / width) * 32 * width + lane * width + i % width;
        float value = 0.0f;
        if (FullAligned || index < orgChunkCount) {
            const unsigned char encoded = encoded_values[i / width][i % width];
            value = tacoDecodeFp8(encoded, fp8_type) * quant_scale;
        }
        values[i] = value;
    }
    tacoWarpHadamard<width>(values);
    float ada_scale = ada_scales[group];
    if (ada_scale < 1e-12f) ada_scale = 1e-12f;
    __align__(16) T output_values[tiles][width];
    #pragma unroll
    for (int i = 0; i < slots; ++i) {
        float value = values[i] * rsqrtf(float(GROUP_SIZE));
        value = check_nan_inf(value) / ada_scale;
        output_values[i / width][i % width] = T(value);
    }
    #pragma unroll
    for (int tile = 0; tile < tiles; ++tile) {
        const size_t first = start + tile * 32 * width + lane * width;
        const int available = FullAligned ? width
            : (first < size_t(orgChunkCount) ? min(width, orgChunkCount - int(first)) : 0);
        tacoStore<T, width, FullAligned>(
            reinterpret_cast<T*>(out_base) + first, output_values[tile], available);
    }
}

template<typename T, int GROUP_SIZE>
void launchCompressTyped(
    const void* input, void* output, int elementsPerChunk,
    size_t inputStride, size_t outputStride, size_t scaleBytes,
    const TacoConfig& config, dim3 grid, cudaStream_t stream) {
    constexpr int threads = GROUP_SIZE == 128 ? 128 : (GROUP_SIZE == 256 ? 64
        : (GROUP_SIZE == 32 ? 512 : 256));
    constexpr size_t rawAlignment = sizeof(T) * GROUP_SIZE / 32 < 16
        ? sizeof(T) * GROUP_SIZE / 32 : 16;
    constexpr size_t encodedAlignment = GROUP_SIZE / 32 < 16
        ? GROUP_SIZE / 32 : 16;
    const bool fullAligned = elementsPerChunk % GROUP_SIZE == 0 &&
        ((reinterpret_cast<uintptr_t>(input) | inputStride) & (rawAlignment - 1)) == 0 &&
        ((reinterpret_cast<uintptr_t>(output) | outputStride) & (encodedAlignment - 1)) == 0;
    const dim3 blocks(DIVUP(grid.x, unsigned(threads / 32)), grid.y);
    const __nv_fp8_interpretation_t format = config.fp8Format == 0 ? __NV_E4M3 : __NV_E5M2;
    const __nv_saturation_t saturation = config.saturate ? __NV_SATFINITE : __NV_NOSAT;
    if (fullAligned) {
        fused_ash_compress_kernel<T, GROUP_SIZE, threads, true><<<blocks, threads, 0, stream>>>(
            input, output, elementsPerChunk, inputStride, outputStride, scaleBytes,
            config.targetRange, config.lambda, config.maxValue(), saturation, format);
    } else {
        fused_ash_compress_kernel<T, GROUP_SIZE, threads, false><<<blocks, threads, 0, stream>>>(
            input, output, elementsPerChunk, inputStride, outputStride, scaleBytes,
            config.targetRange, config.lambda, config.maxValue(), saturation, format);
    }
}

template<typename T, int GROUP_SIZE>
void launchDecompressTyped(
    void* output, const void* input, int elementsPerChunk,
    size_t inputStride, size_t outputStride, size_t scaleBytes,
    const TacoConfig& config, dim3 grid, cudaStream_t stream) {
    constexpr int threads = GROUP_SIZE == 64 ? 512 : 256;
    constexpr size_t rawAlignment = sizeof(T) * GROUP_SIZE / 32 < 16 ? sizeof(T) * GROUP_SIZE / 32 : 16;
    constexpr size_t encodedAlignment = sizeof(T) * GROUP_SIZE / 32 < 16
        ? GROUP_SIZE / 32 : 16 / sizeof(T);
    const bool fullAligned = elementsPerChunk % GROUP_SIZE == 0 &&
        ((reinterpret_cast<uintptr_t>(output) | outputStride) & (rawAlignment - 1)) == 0 &&
        ((reinterpret_cast<uintptr_t>(input) | inputStride) & (encodedAlignment - 1)) == 0;
    const dim3 blocks(DIVUP(grid.x, unsigned(threads / 32)), grid.y);
    const __nv_fp8_interpretation_t format = config.fp8Format == 0 ? __NV_E4M3 : __NV_E5M2;
    if (fullAligned) {
        fused_ash_decompress_kernel<T, GROUP_SIZE, threads, true><<<blocks, threads, 0, stream>>>(
            input, output, elementsPerChunk, inputStride, outputStride, scaleBytes, format);
    } else {
        fused_ash_decompress_kernel<T, GROUP_SIZE, threads, false><<<blocks, threads, 0, stream>>>(
            input, output, elementsPerChunk, inputStride, outputStride, scaleBytes, format);
    }
}

template<int GROUP_SIZE>
void launchCompressKernel(
    ncclDataType_t datatype, const void* input, void* output,
    int elementsPerChunk, size_t inputStride, size_t outputStride,
    size_t scaleBytes, const TacoConfig& config, dim3 grid, cudaStream_t stream) {
    if (datatype == ncclFloat32)
        launchCompressTyped<float, GROUP_SIZE>(input, output, elementsPerChunk,
            inputStride, outputStride, scaleBytes, config, grid, stream);
    else
        launchCompressTyped<__nv_bfloat16, GROUP_SIZE>(input, output, elementsPerChunk,
            inputStride, outputStride, scaleBytes, config, grid, stream);
}

template<int GROUP_SIZE>
void launchDecompressKernel(
    ncclDataType_t datatype, void* output, const void* input,
    int elementsPerChunk, size_t inputStride, size_t outputStride,
    size_t scaleBytes, const TacoConfig& config, dim3 grid, cudaStream_t stream) {
    if (datatype == ncclFloat32)
        launchDecompressTyped<float, GROUP_SIZE>(output, input, elementsPerChunk,
            inputStride, outputStride, scaleBytes, config, grid, stream);
    else
        launchDecompressTyped<__nv_bfloat16, GROUP_SIZE>(output, input, elementsPerChunk,
            inputStride, outputStride, scaleBytes, config, grid, stream);
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
        const size_t inputStride = input.bytes() / input.chunks();
        switch (config.groupSize) {
            case 32:
                launchCompressKernel<32>(
                    input.datatype(), input.data(), output.data(),
                    (int)input.elementsPerChunk(), inputStride,
                    outputChunkBytes, scaleBytes, config, grid,
                    context.stream());
                break;
            case 64:
                launchCompressKernel<64>(
                    input.datatype(), input.data(), output.data(),
                    (int)input.elementsPerChunk(), inputStride,
                    outputChunkBytes, scaleBytes, config, grid,
                    context.stream());
                break;
            case 128:
                launchCompressKernel<128>(
                    input.datatype(), input.data(), output.data(),
                    (int)input.elementsPerChunk(), inputStride,
                    outputChunkBytes, scaleBytes, config, grid,
                    context.stream());
                break;
            case 256:
                launchCompressKernel<256>(
                    input.datatype(), input.data(), output.data(),
                    (int)input.elementsPerChunk(), inputStride,
                    outputChunkBytes, scaleBytes, config, grid,
                    context.stream());
                break;
            case 512:
                launchCompressKernel<512>(
                    input.datatype(), input.data(), output.data(),
                    (int)input.elementsPerChunk(), inputStride,
                    outputChunkBytes, scaleBytes, config, grid,
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
        switch (config.groupSize) {
            case 32:
                launchDecompressKernel<32>(
                    output.datatype(), output.data(), input.data(),
                    (int)elementsPerChunk, inputChunkBytes, outputStride,
                    scaleBytes, config, grid, context.stream());
                break;
            case 64:
                launchDecompressKernel<64>(
                    output.datatype(), output.data(), input.data(),
                    (int)elementsPerChunk, inputChunkBytes, outputStride,
                    scaleBytes, config, grid, context.stream());
                break;
            case 128:
                launchDecompressKernel<128>(
                    output.datatype(), output.data(), input.data(),
                    (int)elementsPerChunk, inputChunkBytes, outputStride,
                    scaleBytes, config, grid, context.stream());
                break;
            case 256:
                launchDecompressKernel<256>(
                    output.datatype(), output.data(), input.data(),
                    (int)elementsPerChunk, inputChunkBytes, outputStride,
                    scaleBytes, config, grid, context.stream());
                break;
            case 512:
                launchDecompressKernel<512>(
                    output.datatype(), output.data(), input.data(),
                    (int)elementsPerChunk, inputChunkBytes, outputStride,
                    scaleBytes, config, grid, context.stream());
                break;
            default: return ncclInvalidArgument;
        }
        return coccl::fromCuda(cudaGetLastError());
    }
};

}  // namespace

COCCL_REGISTER_COMPRESSOR("taco", TacoCompressor);
