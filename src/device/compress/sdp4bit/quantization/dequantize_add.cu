// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

#include "ds_kernel_utils.h"
#include "memory_access_utils.h"
#include "quantization.h"
#include "quantization_utils.h"
#include "reduction_utils.h"
#include "dequantization_utils.h"
// #include <torch/extension.h>

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

template <typename T, int numBits, dequantize::Type qType, int unroll, int threads>
__global__ void dequantize_kernel(
    T* __restrict__ dequant_data,
    const T* model_params, // model params are real params for forward computation
    const int8_t* __restrict__ quant_data,
    // const int64_t* model_param_size, // model param size are used to store size for all params
    // const int64_t total_size,
    const int elems_per_group,
    const int64_t elems_per_chunk,
    const int64_t elems_per_chunk_padding,
    const int num_params,
    const int64_t total_elems)
{
    // extern __shared__ int64_t shared_mem_address[]; // Shared memory declaration

    // Load model_param_size into shared memory
    // int tid = threadIdx.x + threadIdx.y * blockDim.x;
    // for (int i = tid; i < num_params; i += blockDim.x * blockDim.y) {
    //     shared_mem_address[i] = model_param_size[i];
    // }
    // __syncthreads(); // Ensure all threads have loaded the data
    if constexpr (numBits == 4 || numBits == 8) {
        cg::thread_block tb = cg::this_thread_block();
        cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

        // Load constants
        // TODO(cmikeh2): Refactor into functions?
        constexpr int load_granularity = (dequantize::granularity / (sizeof(T))) / (numBits == 8 ? 1 : 2);
        constexpr int load_step_stride = load_granularity * threads;
        constexpr int load_block_stride = load_step_stride * unroll;

        // Store constants
        constexpr int T_per_chunk = dequantize::granularity / sizeof(T);
        constexpr int store_step_stride = T_per_chunk * threads;
        constexpr int store_block_stride = store_step_stride * unroll;
       
        // Load offsets
        const int64_t load_block_offset = tb.group_index().x * load_block_stride;
        // Note: we can use `load_granularity` since the dtype is `int8_t`.
        const int load_thread_offset = tb.thread_index().x * load_granularity;
        
        const int params_size = (sizeof(T) < 4) ? 
                    2 * sizeof(float) : (qType == dequantize::Type::Asymmetric ? 2 : 1) * sizeof(float);

        const int64_t quan_offset_base = ((load_block_offset + load_thread_offset) * (8 / numBits) / elems_per_group + 1) * params_size;

        const int8_t* load_base = quant_data + load_block_offset + load_thread_offset + quan_offset_base;

        // Store offsets
        const int64_t store_block_offset = tb.group_index().x * store_block_stride;
        const int store_thread_offset = tb.thread_index().x * T_per_chunk;
        const int64_t elem_id_base = store_block_offset + store_thread_offset;

        int8_t local_load_buffer[load_granularity * unroll];
        T local_dequant_buffer[T_per_chunk * unroll];
        
        // const int64_t elems_per_chunk_padding = (elems_per_chunk + elems_per_group - 1) / elems_per_group * elems_per_group;

        /*
        Note: Splitting this loop in half gave about 3-5% performance increase for reasons that aren't
        totally clear to me, so this is a deliberately weird code structure.
        */
    #pragma unroll
        for (int i = 0; i < unroll; i++) {
            const int64_t elem_id_iter_padding = elem_id_base + i * store_step_stride;
            const int64_t chunk_id_iter = elem_id_iter_padding / elems_per_chunk_padding;
            const int64_t elem_id_iter_chunk = elem_id_iter_padding % elems_per_chunk_padding;
            const int64_t elem_id_iter_true = chunk_id_iter * elems_per_chunk + elem_id_iter_chunk;
            
            if (elem_id_iter_true < total_elems && elem_id_iter_chunk < elems_per_chunk) {
                // const int64_t quant_offset = ((load_block_offset + load_thread_offset + 
                //     i * load_step_stride) * (8 / numBits) / elems_per_group + 1) * params_size;
                const int64_t quant_step = (i * load_step_stride) * (8 / numBits) / elems_per_group * params_size;

                mem_access::load_global<load_granularity>(local_load_buffer + i * load_granularity,
                                                        load_base + i * load_step_stride + quant_step);
            }
        }

    #pragma unroll
        for (int i = 0; i < unroll; i++) {
            // const int64_t elem_id_iter = elem_id_base + i * store_step_stride;
            const int64_t elem_id_iter_padding = elem_id_base + i * store_step_stride;
            const int64_t chunk_id_iter = elem_id_iter_padding / elems_per_chunk_padding;
            const int64_t elem_id_iter_chunk = elem_id_iter_padding % elems_per_chunk_padding;
            const int64_t elem_id_iter_true = chunk_id_iter * elems_per_chunk + elem_id_iter_chunk;
            
            // const int64_t elem_id_iter_true = (elem_id_iter_padding / elems_per_chunk_padding) * elems_per_chunk + 
            //                                   (elem_id_iter_padding % elems_per_chunk_padding);
            
            if (elem_id_iter_true < total_elems && elem_id_iter_chunk < elems_per_chunk) {
                
                // TODO(cmikeh2): Can we amortize this division? Perform once on the first iteration and
                // use indexing math to do division free interpolation of the successive groups?
                const int64_t group_index = (elem_id_iter_padding / elems_per_group) * (params_size + elems_per_group / (8 / numBits));

                dequantize::Params<qType, numBits> q_params((float*)(quant_data + group_index), 0);

                dequantize::chunk<T, numBits, qType>(local_dequant_buffer + i * T_per_chunk,
                                        local_load_buffer + i * load_granularity,
                                        q_params);
                T temp_param_model[T_per_chunk];
                T* data_cast = local_dequant_buffer + i * T_per_chunk;
                load_to_local<T, T_per_chunk>(
                    temp_param_model, 
                    model_params, 
                    num_params, 
                    total_elems, 
                    elem_id_iter_true);

                for (int k = 0; k < T_per_chunk; k++) {
                    data_cast[k] = __hadd(data_cast[k], temp_param_model[k]);
                }
                mem_access::store_global<dequantize::granularity>(dequant_data + elem_id_iter_true,
                    local_dequant_buffer + i * T_per_chunk);
            }
        }
    } else if constexpr (numBits == 3) {
        // TODO(cmikeh2): Need this implementation
        assert(false);
    } else {
        assert(false);
    }
}

#define LAUNCH_DEQUANT_ADD_KERNEL(num_bits, quant_type)                                          \
    dequantize_kernel<T, num_bits, quant_type, unroll, threads><<<grid, block, 0, stream>>>(     \
        dequant_data, model_params, quant_data, elems_per_group, elems_per_chunk, elems_per_chunk_padding, num_params, total_elems);

template <typename T>
void launch_fused_dequant_add_cuda(
    T* dequant_data,
    const T* model_params,
    const int8_t* quant_data,
    const int num_bits,
    const quantize::Type quant_type,
    const int elems_per_group,
    const int64_t elems_per_chunk,
    const int num_params,
    cudaStream_t stream)
{
    // Calculate total size of the output tensor
   
    // int64_t elems_per_chunk = total_elems / num_params;

    constexpr int unroll = 8;
    constexpr int threads = 256;
    constexpr int elems_per_block = unroll * threads * dequantize::granularity / (sizeof(T));
    const int64_t total_elems = elems_per_chunk * num_params;
    const int64_t elems_per_chunk_padding = (elems_per_chunk + elems_per_group - 1) / elems_per_group * elems_per_group;
    const int64_t total_elems_padding = elems_per_chunk_padding * num_params;

    // int64_t total_elems
    
    const dim3 block(threads);
    const dim3 grid((total_elems_padding + elems_per_block - 1) / elems_per_block);
    // const dim3 grid(1);

    // int64_t shared_mem_size = (num_params) * sizeof(int64_t);

    // TODO(cmikeh2): It may make sense to tune unroll, there is perf benefit for large
    // problem sizes with this large unroll value.
    if (num_bits == 8 && quant_type == quantize::Type::Symmetric) {
        LAUNCH_DEQUANT_ADD_KERNEL(8, quantize::Type::Symmetric);
    } else if (num_bits == 8 && quant_type == quantize::Type::Asymmetric) {
        LAUNCH_DEQUANT_ADD_KERNEL(8, quantize::Type::Asymmetric);
    } else if (num_bits == 4 && quant_type == quantize::Type::Symmetric) {
        LAUNCH_DEQUANT_ADD_KERNEL(4, quantize::Type::Symmetric);
    } else if (num_bits == 4 && quant_type == quantize::Type::Asymmetric) {
        LAUNCH_DEQUANT_ADD_KERNEL(4, quantize::Type::Asymmetric);
    }

}

#ifdef BF16_AVAILABLE

template void launch_fused_dequant_add_cuda(
    __nv_bfloat16* dequant_data,
    const __nv_bfloat16* model_params,
    const int8_t* quant_data,
    const int num_bits,
    const quantize::Type quant_type,
    const int elems_per_group,
    const int64_t elems_per_chunk,
    const int num_params,
    cudaStream_t stream);
#endif