// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <cstdio>
#include "dequantization_utils.h"
#include "ds_kernel_utils.h"
#include "memory_access_utils.h"
#include "quantization_utils.h"
#include "reduction_utils.h"
#include "fast_hadamard_transform.h"
#include "fast_hadamard_transform_common.h"
using rop = reduce::ROpType;

namespace {
constexpr int kMinBlockThreads = 128;
}

/*
TODO(cmikeh2): Add implementation that better handles larger nodes. It would like make sense
to leverage some parallel reductions here to improve performance.
*/

template <typename T, int numBits, int numTensors, int totalChunks,
          quantize::Type quantType>
__global__ void __launch_bounds__(1024) dequant_reduce_ht(T* reduced_data,
                                                       const int8_t* input_data,
                                                       const float* input_scales,
                                                    //    int elems_per_out_group,
                                                       int64_t elems_per_in_tensor,
                                                       int64_t elems_per_in_tensor_padding,
                                                       int groups_per_in_tensor,
                                                       int elems_per_in_group,
                                                       int64_t elements_per_output_chunk,
                                                       int groups_per_output_chunk,
                                                       int num_tensors)
{
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // NOTE(cmikeh2): This probably could be hardcoded to a larger number,
    // but that means even stronger restrictions on the number of elements per group
    // A performance analysis here might be beneficial
    constexpr int storage_values = 16 / sizeof(T);
    constexpr int mem_granularity = storage_values / (8 / numBits);
    constexpr int elems_per_load = mem_granularity / sizeof(int8_t);  // div by 1

    const int group_id = blockIdx.x * blockDim.y + threadIdx.y;
    const bool valid_group = group_id < groups_per_in_tensor;
    const int64_t block_offset = static_cast<int64_t>(group_id) * elems_per_in_group;
    const int elem_offset = tb.thread_index().x * elems_per_load;
    const int64_t base_offset = block_offset + elem_offset;
    const int stride = tb.group_dim().x * elems_per_load;

    T local_buffer[totalChunks * storage_values];

#pragma unroll
    for (int i = 0; i < totalChunks; i++) {
        T* iteration_buffer = local_buffer + i * storage_values;

#pragma unroll
        for (int j = 0; j < storage_values; j++) {
            iteration_buffer[j] = reduce::init<rop::Add, T>();
        }

        const int64_t iter_offset = i * stride + base_offset;
        const int64_t iter_scale_idx = iter_offset / elems_per_in_group;
        const bool do_loads =
            valid_group && i * stride + elem_offset < elems_per_in_group;

        if (numTensors > 0) {
#pragma unroll
            for (int j = 0; j < numTensors; j++) {
                if (do_loads) {
                    int8_t load_buffer[elems_per_load];

                    constexpr int64_t params_size = sizeof(T) < sizeof(float)
                        ? 2 * sizeof(float)
                        : (quantType == quantize::Type::Asymmetric ? 2 : 1) *
                              sizeof(float);
                    const int64_t quant_offset = (j * elems_per_in_tensor_padding + iter_offset) / elems_per_in_group * params_size + params_size; 

                    mem_access::load_global<mem_granularity>(
                        load_buffer, input_data + j * elems_per_in_tensor_padding + iter_offset + quant_offset);

                    const int64_t params_offset = (j * groups_per_in_tensor + iter_scale_idx) * (params_size + elems_per_in_group);

                    // quantize::Params<quantType, numBits> params(
                    //     input_scales + j * groups_per_in_tensor, iter_scale_idx);

                    quantize::Params<quantType, numBits> params(
                            (float*)(input_data + params_offset), 0);

                    T dequant_buffer[storage_values];
                    dequantize::chunk<T, numBits, quantType>(
                        dequant_buffer, load_buffer, params);

#pragma unroll
                    for (int k = 0; k < storage_values; k++) {
                        iteration_buffer[k] =
                            reduce::element<rop::Add>(iteration_buffer[k], dequant_buffer[k]);
                    }
                }
            }
        } else {
#pragma unroll 4
            for (int j = 0; j < num_tensors; j++) {
                if (do_loads) {
                    int8_t load_buffer[elems_per_load];

                    constexpr int64_t params_size = sizeof(T) < sizeof(float)
                        ? 2 * sizeof(float)
                        : (quantType == quantize::Type::Asymmetric ? 2 : 1) *
                              sizeof(float);
                    const int64_t quant_offset = (j * elems_per_in_tensor_padding + iter_offset)  / elems_per_in_group * params_size + params_size; 

                    // mem_access::load_global<mem_granularity>(
                    //     load_buffer, input_data + j * elems_per_in_tensor + iter_offset);
                    
                    mem_access::load_global<mem_granularity>(
                        load_buffer, input_data + j * elems_per_in_tensor_padding + iter_offset + quant_offset);

                    const int64_t params_offset = (j * groups_per_in_tensor + iter_scale_idx) * (params_size + elems_per_in_group);

                    // quantize::Params<quantType, numBits> params(
                    //     input_scales + j * groups_per_in_tensor, iter_scale_idx);
                    quantize::Params<quantType, numBits> params(
                            (float*)(input_data + params_offset), 0);
                    

                    T dequant_buffer[storage_values];
                    dequantize::chunk<T, numBits, quantType>(
                        dequant_buffer, load_buffer, params);

#pragma unroll
                    for (int k = 0; k < storage_values; k++) {
                        iteration_buffer[k] =
                            reduce::element<rop::Add>(iteration_buffer[k], dequant_buffer[k]);
                    }
                }
            }
        }

    }

    // start fixed 32*32 Hadamard Transform, accroding https://github.com/Dao-AILab/fast-hadamard-transform/blob/master/csrc/fast_hadamard_transform_cuda.cu
    constexpr int kNChunks = 1;
    constexpr int kNElts = storage_values;
    constexpr int kLogNElts = cilog2(kNElts);
    constexpr int kNWarps = 32 / kNElts;
    constexpr int kLogWarpSize = cilog2(kNWarps);
#pragma unroll
    for (int i = 0; i < totalChunks; i++) {
        T* iteration_buffer = local_buffer + i * storage_values;
        hadamard_mult_thread_quant<kLogNElts, kNChunks>(iteration_buffer);
        hadamard_mult_warp_quant<kLogWarpSize, 0, kNChunks, kNElts>(iteration_buffer);
        hadamard_inverse_scale_32(iteration_buffer, storage_values);
    }

#pragma unroll
    for (int i = 0; i < totalChunks; i++) {
        const int64_t group_element_offset =
            static_cast<int64_t>(group_id % groups_per_output_chunk) *
            elems_per_in_group * (8 / numBits);
        const int64_t remaining_elements =
            elements_per_output_chunk - group_element_offset;
        const int64_t valid_elements = valid_group
            ? min(static_cast<int64_t>(elems_per_in_group * (8 / numBits)),
                  remaining_elements)
            : 0;
        const int64_t local_element_offset =
            static_cast<int64_t>(i * stride + elem_offset) * (8 / numBits);
        const int64_t output_offset =
            static_cast<int64_t>(group_id / groups_per_output_chunk) *
                elements_per_output_chunk +
            group_element_offset + local_element_offset;
        if (local_element_offset + storage_values <= valid_elements &&
            output_offset % storage_values == 0) {
            mem_access::store_global<16>(reduced_data + output_offset,
                                         local_buffer + i * storage_values);
        } else if (local_element_offset < valid_elements) {
#pragma unroll
            for (int value = 0; value < storage_values; ++value) {
                if (local_element_offset + value < valid_elements) {
                    reduced_data[output_offset + value] =
                        local_buffer[i * storage_values + value];
                }
            }
        }
    }
}

template <int Power>
int32_t pow2_round(int32_t raw_value)
{
    return (((raw_value - 1) >> Power) + 1) << Power;
}

#define LAUNCH_DEQUANT_REDUCE(num_chunks)                      \
    dequant_reduce_ht<T, numBits, numTensors, num_chunks, quantType> \
        <<<grid, block, 0, stream>>>(reduced_data,             \
                                     input_data,               \
                                     input_scales,             \
                                     elems_per_in_tensor,      \
                                     elems_per_in_tensor_padding, \
                                     groups_per_in_tensor,     \
                                     elems_per_in_group,       \
                                     elements_per_output_chunk, \
                                     groups_per_output_chunk,  \
                                     num_tensors);

template <typename T, int numBits, int numTensors,
          quantize::Type quantType>
void launch_dequant_reduce_impl_ht(T* reduced_data,
                                const int8_t* input_data,
                                const float* input_scales,
                                // int out_groups,
                                // int elems_per_out_group,
                                int64_t elems_per_in_tensor,
                                // int groups_per_in_tensor,
                                int elems_per_in_group,
                                int64_t elements_per_output_chunk,
                                int num_tensors,
                                cudaStream_t stream)
{
    constexpr int elems_per_thread =
        (16 / sizeof(T)) / (8 / numBits);
    const int one_step_threads =
        next_pow2((elems_per_in_group + elems_per_thread - 1) / (elems_per_thread));
    // TODO(cmikeh2): Tune this
    const int threads_per_group = (one_step_threads < 1024) ? one_step_threads : 1024;
    // int groups_per_in_tensor = (elems_per_in_tensor + elems_per_in_group - 1) / elems_per_in_group;
    int groups_per_in_tensor = (elems_per_in_tensor * (8 / numBits) + elems_per_in_group * (8 / numBits) - 1) 
                                / (elems_per_in_group * (8 / numBits));
    int64_t elems_per_in_tensor_padding = (int64_t) groups_per_in_tensor * elems_per_in_group;
    int out_groups = groups_per_in_tensor;
    const int64_t elements_per_group =
        static_cast<int64_t>(elems_per_in_group) * (8 / numBits);
    const int groups_per_output_chunk =
        (elements_per_output_chunk + elements_per_group - 1) /
        elements_per_group;

    const int groups_per_block =
        (kMinBlockThreads + threads_per_group - 1) / threads_per_group;
    dim3 block(threads_per_group, groups_per_block);
    dim3 grid((out_groups + groups_per_block - 1) / groups_per_block);

    const int elems_per_step = threads_per_group * elems_per_thread;
    const int unroll_raw = (elems_per_in_group + elems_per_step - 1) / elems_per_step;

    const int unroll = (unroll_raw >= 4) ? pow2_round<1>(unroll_raw) : unroll_raw;

    if (unroll == 1) {
        // 0-4096 elems
        LAUNCH_DEQUANT_REDUCE(1);
    } else if (unroll == 2) {
        // 4097-8192 etc...
        LAUNCH_DEQUANT_REDUCE(2);
    } else if (unroll == 3) {
        LAUNCH_DEQUANT_REDUCE(3);
    } else if (unroll == 4) {
        LAUNCH_DEQUANT_REDUCE(4);
    } else if (unroll == 6) {
        LAUNCH_DEQUANT_REDUCE(6);
    } else if (unroll == 8) {
        LAUNCH_DEQUANT_REDUCE(8);
    } else if (unroll == 10) {
        LAUNCH_DEQUANT_REDUCE(10);
    } else if (unroll == 12) {
        // 48k limit
        LAUNCH_DEQUANT_REDUCE(12);
    } else {
        assert(false);
    }
}

#define LAUNCH_DEQUANT_REDUCE_IMPL(NUM_BITS, NUM_GPUS, QUANT_TYPE)                   \
    launch_dequant_reduce_impl_ht<T, NUM_BITS, NUM_GPUS, QUANT_TYPE>(reduced_data,      \
                                                               input_data,           \
                                                               input_scales,         \
                                                               elems_per_in_tensor,  \
                                                               elems_per_in_group,   \
                                                               elements_per_output_chunk, \
                                                               num_gpus,             \
                                                               stream);

template <typename T>
void launch_dequant_reduce_ht(T* reduced_data,
                           const int8_t* input_data,
                           const float* input_scales,
                           int num_gpus,
                           int num_bits,
                           quantize::Type quant_type,
                        //    int out_groups,
                        //    int elems_per_out_group,
                           int64_t elems_per_in_tensor,
                        //    int groups_per_in_tensor,
                           int elems_per_in_group,
                           int64_t elements_per_output_chunk,
                           cudaStream_t stream)
{
    if (quant_type == quantize::Type::Symmetric) {
        if (num_bits == 4) { 
            if (num_gpus == 1) {
                LAUNCH_DEQUANT_REDUCE_IMPL(4, 1, quantize::Type::Symmetric);
            } else if (num_gpus == 2) {
                LAUNCH_DEQUANT_REDUCE_IMPL(4, 2, quantize::Type::Symmetric);
            } else if (num_gpus == 4) {
                LAUNCH_DEQUANT_REDUCE_IMPL(4, 4, quantize::Type::Symmetric);
            } else if (num_gpus == 8) {
                LAUNCH_DEQUANT_REDUCE_IMPL(4, 8, quantize::Type::Symmetric);
            } else if (num_gpus == 16) {
                LAUNCH_DEQUANT_REDUCE_IMPL(4, 16, quantize::Type::Symmetric);
            } else {
                LAUNCH_DEQUANT_REDUCE_IMPL(4, -1, quantize::Type::Symmetric);
            }
        } else if (num_bits == 8) {
            if (num_gpus == 1) {
                LAUNCH_DEQUANT_REDUCE_IMPL(8, 1, quantize::Type::Symmetric);
            } else if (num_gpus == 2) {
                LAUNCH_DEQUANT_REDUCE_IMPL(8, 2, quantize::Type::Symmetric);
            } else if (num_gpus == 4) {
                LAUNCH_DEQUANT_REDUCE_IMPL(8, 4, quantize::Type::Symmetric);
            } else if (num_gpus == 8) {
                LAUNCH_DEQUANT_REDUCE_IMPL(8, 8, quantize::Type::Symmetric);
            } else if (num_gpus == 16) {
                LAUNCH_DEQUANT_REDUCE_IMPL(8, 16, quantize::Type::Symmetric);
            } else {
                LAUNCH_DEQUANT_REDUCE_IMPL(8, -1, quantize::Type::Symmetric);
            }
        }
    } else if (quant_type == quantize::Type::Asymmetric) {
        if (num_bits == 4) {
            if (num_gpus == 8) {
                LAUNCH_DEQUANT_REDUCE_IMPL(4, 8, quantize::Type::Asymmetric);
            } else if (num_gpus == 16) {
                LAUNCH_DEQUANT_REDUCE_IMPL(4, 16, quantize::Type::Asymmetric);
            } else {
                LAUNCH_DEQUANT_REDUCE_IMPL(4, -1, quantize::Type::Asymmetric);
            }
        } else if (num_bits == 8) {
            if (num_gpus == 8) {
                LAUNCH_DEQUANT_REDUCE_IMPL(8, 8, quantize::Type::Asymmetric);
            } else if (num_gpus == 16) {
                LAUNCH_DEQUANT_REDUCE_IMPL(8, 16, quantize::Type::Asymmetric);
            } else {
                LAUNCH_DEQUANT_REDUCE_IMPL(8, -1, quantize::Type::Asymmetric);
            }
        }
    }
}

template void launch_dequant_reduce_ht<float>(
    float*, const int8_t*, const float*, int, int, quantize::Type, int64_t,
    int, int64_t, cudaStream_t);
template void launch_dequant_reduce_ht<__half>(
    __half*, const int8_t*, const float*, int, int, quantize::Type, int64_t,
    int, int64_t, cudaStream_t);
#ifdef BF16_AVAILABLE
template void launch_dequant_reduce_ht<__nv_bfloat16>(
    __nv_bfloat16*, const int8_t*, const float*, int, int, quantize::Type,
    int64_t, int, int64_t, cudaStream_t);
#endif
