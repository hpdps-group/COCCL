// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

#include "ds_kernel_utils.h"
#include "memory_access_utils.h"
#include "quantization.h"
#include "quantization_utils.h"
#include "reduction_utils.h"
// #include <torch/extension.h>
// #include <ATen/ATen.h>
// #include <ATen/cuda/CUDAContext.h>
namespace cg = cooperative_groups;

template <typename scalar_t, int vec_size>
__device__ __inline__ void load_to_local(
    scalar_t* __restrict__ local_buffer,
    const scalar_t* __restrict__ model_params,
    const int num_params,
    const int64_t total_length,
    const int64_t idx) {

    // int64_t left = param_offset;
    // int64_t right = num_params - 1;
    // int64_t param_idx = num_params;

    // binary search for param list offset
    // while (left <= right) {
    //     int64_t mid = (left + right) / 2;
    //     int64_t mid_start_idx = mid==0 ? 0 : param_sizes[mid-1];
    //     int64_t mid_end_idx = param_sizes[mid];
    //     if (mid_end_idx <= idx) {
    //         left = mid + 1;
    //     } else if (idx < mid_start_idx) {
    //         right = mid - 1;
    //     } else {
    //         param_idx = mid;
    //         break;
    //     }
    // }
    int64_t chunk_length = total_length / num_params;
    int64_t param_idx = idx / chunk_length;

    if (param_idx >= num_params) {

#pragma unroll
        for (int j = 0; j < vec_size; j++) {
            local_buffer[j] = 0;
        }

        return ;
    }
    

#pragma unroll
    for (int j = 0; j < vec_size; ) {
        if (idx + j >= total_length) {
            local_buffer[j] = 0; // Handle out-of-bounds by setting to zero or another appropriate value
            j++;
            continue;
        }

        int64_t start_idx = param_idx==0 ? 0 : param_idx * chunk_length;
        int64_t end_idx = (param_idx + 1) * chunk_length;
        assert(("load_to_local failed, idx + j < start_idx", start_idx <= idx + j));
        for (; param_idx < num_params; ) {
            int64_t param_offset = param_idx * chunk_length;
            if (idx + j < end_idx) {
                if (idx + vec_size - 1 < end_idx) {
                    // IF [idx+j, idx+vec_size) is contiguous, load once is enough
                    // model_params+ param_offset [idx + j - start_idx: idx + vec_size] -> local_buffer[j: vec_size]
                    if (vec_size - j >= 8) {
                        mem_access::load_global<8*sizeof(scalar_t)>(
                            local_buffer + j,
                            model_params + param_offset +idx + j - start_idx);
                        j += 8;
                        break;
                    } else if (vec_size - j >= 4) {
                        mem_access::load_global<4*sizeof(scalar_t)>(
                            local_buffer + j,
                            model_params+ param_offset +idx + j - start_idx);
                        j += 4;
                        break;
                    } else if (vec_size - j >= 2) {
                        mem_access::load_global<2*sizeof(scalar_t)>(
                            local_buffer + j,
                            model_params+ param_offset +idx + j - start_idx);
                        j += 2;
                        break;
                    } else if (vec_size - j >= 1) {
                        mem_access::load_global<1*sizeof(scalar_t)>(
                            local_buffer + j,
                            model_params+ param_offset +idx + j - start_idx);
                        j += 1;
                        break;
                    } else {
                        assert(("load_to_local failed, vec_size - j < 1", false));
                    }
                } else {
                    // IF [idx+j, idx+vec_size) is not contiguous, only load [idx+j, end_idx)
                    if (end_idx - idx - j >= 8) {
                        mem_access::load_global<8*sizeof(scalar_t)>(
                            local_buffer + j,
                            model_params+ param_offset +idx + j - start_idx);
                        j += 8;
                        break;
                    } else if (end_idx - idx - j >= 4) {
                        mem_access::load_global<4*sizeof(scalar_t)>(
                            local_buffer + j,
                            model_params+ param_offset +idx + j - start_idx);
                        j += 4;
                        break;
                    } else if (end_idx - idx - j >= 2) {
                        mem_access::load_global<2*sizeof(scalar_t)>(
                            local_buffer + j,
                            model_params+ param_offset +idx + j - start_idx);
                        j += 2;
                        break;
                    } else if (end_idx - idx - j >= 1) {
                        mem_access::load_global<1*sizeof(scalar_t)>(
                            local_buffer + j,
                            model_params+ param_offset +idx + j - start_idx);
                        j += 1;
                        break;
                    } else {
                        assert(("load_to_local failed, end_idx - idx - j < 1", false));
                    }
                }
            }
            start_idx = end_idx;
            ++param_idx;
            end_idx += chunk_length;
        }
        assert(("load_to_local failed, for loop search parameters finished without finding", param_idx<num_params));
    }
    // return param_idx;
}

#ifdef BF16_AVAILABLE
template <
    int q_bits,
    quantize::Type quant_type,
    int UNROLL,
    int internal_unroll,
    int threads_per_group,
    int max_threads>
__global__ void cached_quantization(
    int8_t* __restrict__ output_data,
    const __nv_bfloat16* model_params, // model params are real params for forward computation
    const __nv_bfloat16* __restrict__ shard_params_buffer, // Updated to bfloat16
    int groups,
    int elems_per_group,
    int64_t elems_per_chunk,
    int64_t elems_per_chunk_padding,
    const int num_params,
    const int64_t dp_param_offset,
    const int64_t total_elems)
{
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // Indexing offsets
    const int64_t block_offset =
        (static_cast<int64_t>(tb.group_index().x) * (max_threads / threads_per_group) * elems_per_group) +
        (tb.thread_index().y * elems_per_group);
    const int elem_offset_in_group = tb.thread_index().x * 8;
    const int64_t padding_offset = block_offset + elem_offset_in_group;
    const int64_t chunk_idx = padding_offset / elems_per_chunk_padding;
    const int64_t elem_offset_in_chunk = padding_offset % elems_per_chunk_padding;
    const int64_t base_offset = chunk_idx * elems_per_chunk + elem_offset_in_chunk;

    const int stride = tb.size() * 8;

    const __nv_bfloat16* input_base = shard_params_buffer + base_offset;
    // __nv_bfloat16* output_base = output_data + base_offset;
    // printf("base_offset %ld elem_offset %d\n", base_offset, elem_offset);
    __nv_bfloat162 local_buffer[UNROLL * internal_unroll * 4]; // Updated buffer type
    // int64_t param_offset = d_block_start_param_offset[tb.group_index().x];
    // int64_t param_offset = 0;

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
        __nv_bfloat162* iteration_buffer = local_buffer + i * internal_unroll * 4; // Updated pointer type
#pragma unroll
        for (int j = 0; j < internal_unroll; j++) {
            const int iteration = i * internal_unroll + j;
            __nv_bfloat16* data_cast = reinterpret_cast<__nv_bfloat16*>(iteration_buffer + j * 4);
            __nv_bfloat16 temp_param_model[8];
            mem_access::load_global<16>(
                iteration_buffer + j * 4,
                input_base + iteration * stride,
                (elem_offset_in_group + iteration * stride < elems_per_group) &&
                (elem_offset_in_chunk + iteration * stride < elems_per_chunk));            
            load_to_local<__nv_bfloat16, 8>(temp_param_model, model_params, num_params, total_elems, base_offset + iteration * stride + dp_param_offset);

#pragma unroll
            for (int k = 0; k < 8; k++) {
                data_cast[k] = ((elem_offset_in_group + iteration * stride + k < elems_per_group) &&
                                (elem_offset_in_chunk + iteration * stride + k < elems_per_chunk)) ? __hsub(data_cast[k], temp_param_model[k]) : __float2bfloat16(0.0f);
            }
        }
    }

    quantize::
        local_array<quant_type, q_bits, UNROLL * internal_unroll, threads_per_group, max_threads>(
            local_buffer, nullptr, output_data, elems_per_group, groups);
}
#endif


// #ifdef BF16_AVAILABLE
// void launch_sub(
//     int8_t* __restrict__ output_data,
//     const __nv_bfloat16* d_param_buffer,  // param buffer are contiguous buffer place for all gather params
//     const int64_t param_buffer_size,
//     const __nv_bfloat16* d_model_params,
//     const int64_t dp_param_offset,
//     const int groups,
//     const int elems_per_group,
//     const int num_params,
//     const int num_bits,
//     quantize::Type quant_type, 
//     cudaStream_t stream)
// {
//     constexpr int max_threads = 256;
//     constexpr int internal_unroll = 2;
//     const bool is_subblock_schedule = (elems_per_group <= 128) ? true : false;
//     const int bf_per_step = is_subblock_schedule ? 8
//                                                 : 8 * internal_unroll;

//     const int one_step_threads = next_pow2((elems_per_group + bf_per_step - 1) / bf_per_step);
//     const int threads_per_group = (one_step_threads < max_threads) ? one_step_threads : max_threads;
//     const int groups_per_block = is_subblock_schedule ? min((max_threads + threads_per_group - 1) / threads_per_group, groups) : 1;
//     const int groups_launch = (groups_per_block + groups - 1) / groups_per_block;

//     // printf("threads_per_group: %d, groups_per_block: %d, groups_launch: %d\n", threads_per_group, groups_per_block, groups_launch);

//     dim3 block(threads_per_group, groups_per_block);
//     dim3 grid(groups_launch);
//     const int elems_per_step = threads_per_group * bf_per_step;
//     const int external_unroll = (elems_per_group + elems_per_step - 1) / elems_per_step;

//     // Calculate total size of the output tensor
//     int64_t total_size = 0;
//     // int64_t num_params = param_list.size();
//     // std::vector<int64_t> model_param_size(num_params);
//     // for (int64_t i = 0; i < num_params; ++i) {
//     //     total_size += param_list[i].size(0);
//     //     model_param_size[i] = total_size;
//     // }
//     // // printf("total size: %ld, param buffer size: %ld, dp_param_offset: %ld\n", total_size, param_buffer_size, dp_param_offset);
//     // total_size = min(total_size, param_buffer_size+dp_param_offset);
//     for (int64_t i = 0; i < num_params; ++i) {
//         total_size += groups * elems_per_group;
//     }
//     total_size = min(total_size, param_buffer_size+dp_param_offset);
//     // Copy params ptr
//     // std::vector<void*> model_params(num_params);
//     // for (int64_t i = 0; i < num_params; ++i) {
//     //     model_params[i] = param_list[i].data_ptr();
//     // }

//     // // Allocate device memory for input pointers
//     // __nv_bfloat16** d_model_params;
//     // cudaMalloc(&d_model_params, param_list.size() * sizeof(__nv_bfloat16*));
//     // cudaMemcpy(d_model_params, model_params.data(), param_list.size() * sizeof(__nv_bfloat16*), cudaMemcpyHostToDevice);

//     // // Allocate device memory for input sizes
//     // int64_t* d_model_param_size;
//     // cudaMalloc(&d_model_param_size, (num_params) * sizeof(int64_t));
//     // cudaMemcpy(d_model_param_size, model_param_size.data(), num_params * sizeof(int64_t), cudaMemcpyHostToDevice);


//     // int64_t shared_mem_size = (num_params) * sizeof(int64_t);

//     if (is_subblock_schedule) {
//         if (threads_per_group == 1) {
//             LAUNCH_CACHED_QUANT(num_bits, quant_type, 1, 1, 1);
//         } else if (threads_per_group == 2) {
//             LAUNCH_CACHED_QUANT(num_bits, quant_type, 1, 1, 2);
//         } else if (threads_per_group == 4) {
//             LAUNCH_CACHED_QUANT(num_bits, quant_type, 1, 1, 4);
//         } else if (threads_per_group == 8) {
//             LAUNCH_CACHED_QUANT(num_bits, quant_type, 1, 1, 8);
//         } else if (threads_per_group == 16) {
//             LAUNCH_CACHED_QUANT(num_bits, quant_type, 1, 1, 16);
//         }
//     } else if (external_unroll == 1) {
//         LAUNCH_CACHED_QUANT(num_bits, quant_type, 1, internal_unroll, max_threads);
//     } else if (external_unroll == 2) {
//         LAUNCH_CACHED_QUANT(num_bits, quant_type, 2, internal_unroll, max_threads);
//     } else if (external_unroll == 3) {
//         LAUNCH_CACHED_QUANT(num_bits, quant_type, 3, internal_unroll, max_threads);
//     } else if (external_unroll == 4) {
//         LAUNCH_CACHED_QUANT(num_bits, quant_type, 4, internal_unroll, max_threads);
//     }
// }
// #endif

/********* Launcher methods ***********/
#define LAUNCH_CACHED_QUANT_CALL(q_bits, quant_type)                            \
    cached_quantization<q_bits,                      \
                        quant_type,                  \
                        unroll_factor,               \
                        internal_unroll_l,           \
                        threads_per_group,           \
                        max_threads>                 \
        <<<grid, block, 0, stream>>>(output_data, param_list, shard_params_buffer, groups, elems_per_group, elems_per_chunk, elems_per_chunk_padding, num_params, dp_param_offset, total_elems);

#define LAUNCH_CACHED_QUANT(                                                                 \
    q_bits, quant_type, unroll_factor_in, internal_unroll_in, threads_per_group_in)  \
    const int unroll_factor = unroll_factor_in;                                     \
    const int internal_unroll_l = internal_unroll_in;                               \
    const int threads_per_group = threads_per_group_in;                             \
    if (q_bits == 4) {                                                              \
        if (quant_type == quantize::Type::Asymmetric) {                             \
            LAUNCH_CACHED_QUANT_CALL(4, quantize::Type::Asymmetric)                 \
        } else {                                                                    \
            LAUNCH_CACHED_QUANT_CALL(4, quantize::Type::Symmetric)                  \
        }                                                                           \
    } else {                                                                        \
        if (quant_type == quantize::Type::Asymmetric) {                             \
            LAUNCH_CACHED_QUANT_CALL(8, quantize::Type::Asymmetric)                 \
        } else {                                                                    \
            LAUNCH_CACHED_QUANT_CALL(8, quantize::Type::Symmetric)                  \
        }                                                                           \
    }

void launch_fused_sub_quant_cuda(
    int8_t* output_data,
    const __nv_bfloat16* param_list,
    const __nv_bfloat16* shard_params_buffer,
    const int num_bits,
    const quantize::Type quant_type,
    // const int groups,
    const int elems_per_group,
    const int64_t elems_per_chunk,
    const int num_params,
    const int chunk_offset,
    cudaStream_t stream)
{
    constexpr int max_threads = 256;
    constexpr int internal_unroll = 2;
    const bool is_subblock_schedule = (elems_per_group <= 128) ? true : false;
    const int bf_per_step = is_subblock_schedule ? 8
                                                : 8 * internal_unroll;

    const int one_step_threads = next_pow2((elems_per_group + bf_per_step - 1) / bf_per_step);
    const int threads_per_group = (one_step_threads < max_threads) ? one_step_threads : max_threads;

    const int groups_per_chunk = (elems_per_chunk + elems_per_group -1) / elems_per_group;
    const int64_t elems_per_chunk_padding = (int64_t)groups_per_chunk * elems_per_group;
    const int groups = groups_per_chunk;

    const int groups_per_block = is_subblock_schedule ? min((max_threads + threads_per_group - 1) / threads_per_group, groups) : 1;
    const int groups_launch = (groups_per_block + groups - 1) / groups_per_block;

    // printf("threads_per_group: %d, groups_per_block: %d, groups_launch: %d\n", threads_per_group, groups_per_block, groups_launch);

    dim3 block(threads_per_group, groups_per_block);
    dim3 grid(groups_launch);
    const int elems_per_step = threads_per_group * bf_per_step;
    const int external_unroll = (elems_per_group + elems_per_step - 1) / elems_per_step;

    // Calculate total size of the output tensor
    int64_t total_elems = elems_per_chunk * num_params;

    // int64_t num_params = param_list.size();
    // std::vector<int64_t> model_param_size(num_params);
    // for (int64_t i = 0; i < num_params; ++i) {
    //     total_size += param_list[i].size(0);
    //     model_param_size[i] = total_size;
    // }
    // // printf("total size: %ld, param buffer size: %ld, dp_param_offset: %ld\n", total_size, param_buffer_size, dp_param_offset);
    // total_size = min(total_size, param_buffer_size+dp_param_offset);
    // for (int64_t i = 0; i < num_params; ++i) {
    //     total_size += groups * elems_per_group;
    // }
    const int64_t dp_param_offset = chunk_offset * elems_per_chunk;
    total_elems = min(total_elems, elems_per_chunk+dp_param_offset);

    if (is_subblock_schedule) {
        if (threads_per_group == 1) {
            LAUNCH_CACHED_QUANT(num_bits, quant_type, 1, 1, 1);
        } else if (threads_per_group == 2) {
            LAUNCH_CACHED_QUANT(num_bits, quant_type, 1, 1, 2);
        } else if (threads_per_group == 4) {
            LAUNCH_CACHED_QUANT(num_bits, quant_type, 1, 1, 4);
        } else if (threads_per_group == 8) {
            LAUNCH_CACHED_QUANT(num_bits, quant_type, 1, 1, 8);
        } else if (threads_per_group == 16) {
            LAUNCH_CACHED_QUANT(num_bits, quant_type, 1, 1, 16);
        }
    } else if (external_unroll == 1) {
        LAUNCH_CACHED_QUANT(num_bits, quant_type, 1, internal_unroll, max_threads);
    } else if (external_unroll == 2) {
        LAUNCH_CACHED_QUANT(num_bits, quant_type, 2, internal_unroll, max_threads);
    } else if (external_unroll == 3) {
        LAUNCH_CACHED_QUANT(num_bits, quant_type, 3, internal_unroll, max_threads);
    } else if (external_unroll == 4) {
        LAUNCH_CACHED_QUANT(num_bits, quant_type, 4, internal_unroll, max_threads);
    }


    // // auto output_options = at::TensorOptions()
    // //                           .dtype(at::ScalarType::BFloat16)
    // //                           .layout(at::kStrided)
    // //                           .device(at::kCUDA)
    // //                           .requires_grad(false);

    // // auto output_sizes = input_vals.sizes();
    // // output_sizes[output_sizes.size() - 1] /= numBits == 8 ? 1 : 2;
    // // auto output = torch::empty_like(input_vals);
    // // const int elems_per_group = at::numel(input_vals) < 2048 ? at::numel(input_vals) : 2048;
    // // const int groups = at::numel(input_vals) / elems_per_group;
    // // const int elems_per_group = at::numel(input_vals) / groups;
    // // if (input_vals.scalar_type() == at::ScalarType::Half) {
    // //     // launch_quant((int8_t*)output.data_ptr(),
    // //     //             (float*)params.data_ptr(),
    // //     //             (__half*)input_vals.data_ptr(),
    // //     //             groups,
    // //     //             elems_per_group,
    // //     //             numBits,
    // //     //             quantType,
    // //     //             at::cuda::getCurrentCUDAStream());
    // //     // return {output, params};
    // //     throw std::runtime_error("Unsupported input tensor data type.");
    // // } else if (input_vals.scalar_type() == at::ScalarType::Float) {
    // //     // launch_quant((int8_t*)output.data_ptr(),
    // //     //             (float*)params.data_ptr(),
    // //     //             (float*)input_vals.data_ptr(),
    // //     //             groups,
    // //     //             elems_per_group,
    // //     //             numBits,
    // //     //             quantType,
    // //     //             at::cuda::getCurrentCUDAStream());
    // //     // return {output, params};
    // //     throw std::runtime_error("Unsupported input tensor data type.");
    // // } else if (input_vals.scalar_type() == at::ScalarType::BFloat16) {
    // //     // launch_sub(
    // //     //     (__nv_bfloat16*)output_buffer.data_ptr(),
    // //     //     (__nv_bfloat16*)input_vals.data_ptr(),
    // //     //     at::numel(input_vals),
    // //     //     param_list,
    // //     //     dp_param_offset,
    // //     //     groups,
    // //     //     elems_per_group,
    //     //     at::cuda::getCurrentCUDAStream());
    //     launch_sub(
    //         (int8_t*)output_buffer,
    //         (__nv_bfloat16*)input_vals,
    //         groups * elems_per_group,
    //         (__nv_bfloat16*)param_list,
    //         dp_param_offset,
    //         groups,
    //         elems_per_group,
    //         num_params,
    //         num_bits,
    //         quant_type,
    //         stream);
    //     // return output_buffer;
    // // } else {
    // //     throw std::runtime_error("Unsupported input tensor data type.");
    // // }
}