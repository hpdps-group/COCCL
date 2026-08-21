// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

#include "ds_kernel_utils.h"
#include "memory_access_utils.h"
#include "quantization.h"
// #include "quantization_utils.h"
// #include "reduction_utils.h"
// #include <torch/extension.h>
// #include <ATen/ATen.h>
// #include <ATen/cuda/CUDAContext.h>
namespace cg = cooperative_groups;

template <typename scalar_t, int vec_size>
__device__ __inline__ size_t load_to_local(
    scalar_t* __restrict__ local_buffer,
    const scalar_t* __restrict__ model_params,
    const size_t num_params,
    const size_t total_length,
    const size_t idx) {

    // size_t left = param_offset;
    // size_t right = num_params - 1;
    // size_t param_idx = num_params;

    // binary search for param list offset
    // while (left <= right) {
    //     size_t mid = (left + right) / 2;
    //     size_t mid_start_idx = mid==0 ? 0 : param_sizes[mid-1];
    //     size_t mid_end_idx = param_sizes[mid];
    //     if (mid_end_idx <= idx) {
    //         left = mid + 1;
    //     } else if (idx < mid_start_idx) {
    //         right = mid - 1;
    //     } else {
    //         param_idx = mid;
    //         break;
    //     }
    // }
    size_t chunk_length = total_length / num_params;
    size_t param_idx = idx / chunk_length;

    if (param_idx >= num_params) {

#pragma unroll
        for (int j = 0; j < vec_size; j++) {
            local_buffer[j] = 0;
        }

        return num_params;
    }
    

#pragma unroll
    for (int j = 0; j < vec_size; ) {
        if (idx + j >= total_length) {
            local_buffer[j] = 0; // Handle out-of-bounds by setting to zero or another appropriate value
            j++;
            continue;
        }

        size_t start_idx = param_idx==0 ? 0 : param_idx * chunk_length;
        size_t end_idx = (param_idx + 1) * chunk_length;
        assert(("load_to_local failed, idx + j < start_idx", start_idx <= idx + j));
        for (; param_idx < num_params; ) {
            size_t param_offset = param_idx * chunk_length;
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
    int UNROLL,
    int internal_unroll,
    int threads_per_group,
    int max_threads>
__global__ void fused_sub_kernel(
    __nv_bfloat16* __restrict__ output_data,
    const __nv_bfloat16* __restrict__ param_buffer, // Updated to bfloat16
    const __nv_bfloat16* model_params, // model params are real params for forward computation
    const size_t dp_param_offset,
    const size_t num_params,
    const size_t total_size,
    int groups,
    int elems_per_group)
{
    // printf("UNROLL %d internal_unroll %d threads_per_group %d\n", UNROLL, internal_unroll, threads_per_group);
    extern __shared__ size_t shared_mem_address[]; // Shared memory declaration

    // Load model_param_size into shared memory
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    // for (int i = tid; i < num_params; i += blockDim.x * blockDim.y) {
    //     shared_mem_address[i] = model_param_size[i];
    // }
    // __syncthreads(); // Ensure all threads have loaded the data

    // size_t* shared_model_param_size = shared_mem_address;

    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // Indexing offsets
    const int64_t block_offset =
        (static_cast<int64_t>(tb.group_index().x) * (max_threads / threads_per_group) * elems_per_group) +
        (tb.thread_index().y * elems_per_group);
    const int elem_offset = tb.thread_index().x * 8;
    const int64_t base_offset = block_offset + elem_offset;
    const int stride = tb.size() * 8;

    const __nv_bfloat16* input_base = param_buffer + base_offset;
    __nv_bfloat16* output_base = output_data + base_offset;
    // printf("base_offset %ld elem_offset %d\n", base_offset, elem_offset);
    __nv_bfloat162 local_buffer[UNROLL * internal_unroll * 4]; // Updated buffer type
    // size_t param_offset = d_block_start_param_offset[tb.group_index().x];
    // size_t param_offset = 0;

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
        __nv_bfloat162* iteration_buffer = local_buffer + i * internal_unroll * 4; // Updated pointer type
#pragma unroll
        for (int j = 0; j < internal_unroll; j++) {
            const int iteration = i * internal_unroll + j;
            __nv_bfloat16* data_cast = reinterpret_cast<__nv_bfloat16*>(iteration_buffer + j * 4);
            __nv_bfloat16 temp_param_model[8];
            // if(elem_offset + iteration * stride < elems_per_group)
            //    printf("idx %d input %.5f\n",base_offset +iteration * stride, __bfloat162float( *(input_base + iteration * stride)));
            mem_access::load_global<16>(
                iteration_buffer + j * 4,
                input_base + iteration * stride,
                elem_offset + iteration * stride < elems_per_group);

            
            // param_offset = load_to_local<__nv_bfloat16, 8>(temp_param_model, model_params, shared_model_param_size, num_params, total_size, base_offset + iteration * stride + dp_param_offset, param_offset);
            load_to_local<__nv_bfloat16, 8>(temp_param_model, model_params, num_params, total_size, base_offset + iteration * stride + dp_param_offset);

#pragma unroll
            for (int k = 0; k < 8; k++) {
                // if(elem_offset + iteration * stride + k < elems_per_group)
                //     printf("idx %d sub %.5f - %.5f\n",block_offset + elem_offset + iteration * stride + k, __bfloat162float(data_cast[k]), __bfloat162float(temp_param_model[k]));
                data_cast[k] = (elem_offset + iteration * stride + k < elems_per_group) ? __hsub(data_cast[k], temp_param_model[k]) : __float2bfloat16(0.0f);
            }
            if(elem_offset + iteration * stride < elems_per_group){
                mem_access::store_global<16>(
                    output_base + iteration * stride,
                    iteration_buffer + j * 4
                );
            //     // if(block_offset + elem_offset + iteration * stride < total_size) 
            //     // printf("threadIdx %d idx %d\n",tid,  block_offset + elem_offset + iteration * stride);
            }
        }
    }

}
#endif

/********* Launcher methods ***********/
#define LAUNCH_SUB_CALL()                            \
    fused_sub_kernel<unroll_factor,                  \
                        internal_unroll_l,           \
                        threads_per_group,           \
                        max_threads>                 \
        <<<grid, block, 0, stream>>>(output_data, d_param_buffer, d_model_params, dp_param_offset, num_params, total_size, groups, elems_per_group);

#define LAUNCH_SUB(                                                                 \
    unroll_factor_in, internal_unroll_in, threads_per_group_in)                     \
    const int unroll_factor = unroll_factor_in;                                     \
    const int internal_unroll_l = internal_unroll_in;                               \
    const int threads_per_group = threads_per_group_in;                             \
    LAUNCH_SUB_CALL()

#ifdef BF16_AVAILABLE
void launch_sub(
    __nv_bfloat16* __restrict__ output_data,
    const __nv_bfloat16* d_param_buffer,  // param buffer are contiguous buffer place for all gather params
    const size_t param_buffer_size,
    const __nv_bfloat16* d_model_params,
    const size_t dp_param_offset,
    const int groups,
    const int elems_per_group,
    const size_t num_params,
    cudaStream_t stream)
{
    constexpr int max_threads = 256;
    constexpr int internal_unroll = 2;
    const bool is_subblock_schedule = (elems_per_group <= 128) ? true : false;
    const int bf_per_step = is_subblock_schedule ? 8
                                                : 8 * internal_unroll;

    const int one_step_threads = next_pow2((elems_per_group + bf_per_step - 1) / bf_per_step);
    const int threads_per_group = (one_step_threads < max_threads) ? one_step_threads : max_threads;
    const int groups_per_block = is_subblock_schedule ? min((max_threads + threads_per_group - 1) / threads_per_group, groups) : 1;
    const int groups_launch = (groups_per_block + groups - 1) / groups_per_block;

    // printf("threads_per_group: %d, groups_per_block: %d, groups_launch: %d\n", threads_per_group, groups_per_block, groups_launch);

    dim3 block(threads_per_group, groups_per_block);
    dim3 grid(groups_launch);
    const int elems_per_step = threads_per_group * bf_per_step;
    const int external_unroll = (elems_per_group + elems_per_step - 1) / elems_per_step;

    // Calculate total size of the output tensor
    size_t total_size = 0;
    // size_t num_params = param_list.size();
    // std::vector<size_t> model_param_size(num_params);
    // for (size_t i = 0; i < num_params; ++i) {
    //     total_size += param_list[i].size(0);
    //     model_param_size[i] = total_size;
    // }
    // // printf("total size: %ld, param buffer size: %ld, dp_param_offset: %ld\n", total_size, param_buffer_size, dp_param_offset);
    // total_size = min(total_size, param_buffer_size+dp_param_offset);
    for (size_t i = 0; i < num_params; ++i) {
        total_size += groups * elems_per_group;
    }
    total_size = min(total_size, param_buffer_size+dp_param_offset);
    // Copy params ptr
    // std::vector<void*> model_params(num_params);
    // for (size_t i = 0; i < num_params; ++i) {
    //     model_params[i] = param_list[i].data_ptr();
    // }

    // // Allocate device memory for input pointers
    // __nv_bfloat16** d_model_params;
    // cudaMalloc(&d_model_params, param_list.size() * sizeof(__nv_bfloat16*));
    // cudaMemcpy(d_model_params, model_params.data(), param_list.size() * sizeof(__nv_bfloat16*), cudaMemcpyHostToDevice);

    // // Allocate device memory for input sizes
    // size_t* d_model_param_size;
    // cudaMalloc(&d_model_param_size, (num_params) * sizeof(size_t));
    // cudaMemcpy(d_model_param_size, model_param_size.data(), num_params * sizeof(size_t), cudaMemcpyHostToDevice);


    // size_t shared_mem_size = (num_params) * sizeof(size_t);

    if (is_subblock_schedule) {
        if (threads_per_group == 1) {
            LAUNCH_SUB(1, 1, 1);
        } else if (threads_per_group == 2) {
            LAUNCH_SUB(1, 1, 2);
        } else if (threads_per_group == 4) {
            LAUNCH_SUB(1, 1, 4);
        } else if (threads_per_group == 8) {
            LAUNCH_SUB(1, 1, 8);
        } else if (threads_per_group == 16) {
            LAUNCH_SUB(1, 1, 16);
        }
    } else if (external_unroll == 1) {
        LAUNCH_SUB(1, internal_unroll, max_threads);
    } else if (external_unroll == 2) {
        LAUNCH_SUB(2, internal_unroll, max_threads);
    } else if (external_unroll == 3) {
        LAUNCH_SUB(3, internal_unroll, max_threads);
    } else if (external_unroll == 4) {
        LAUNCH_SUB(4, internal_unroll, max_threads);
    }
}
#endif


void launch_fused_sub_cuda(
    __nv_bfloat16* output_buffer,
    const __nv_bfloat16* input_vals,
    const __nv_bfloat16* param_list,
    const int groups,
    const int elems_per_group,
    const size_t num_params,
    const size_t dp_param_offset,
    cudaStream_t stream) {


    // auto output_options = at::TensorOptions()
    //                           .dtype(at::ScalarType::BFloat16)
    //                           .layout(at::kStrided)
    //                           .device(at::kCUDA)
    //                           .requires_grad(false);

    // auto output_sizes = input_vals.sizes();
    // output_sizes[output_sizes.size() - 1] /= numBits == 8 ? 1 : 2;
    // auto output = torch::empty_like(input_vals);
    // const int elems_per_group = at::numel(input_vals) < 2048 ? at::numel(input_vals) : 2048;
    // const int groups = at::numel(input_vals) / elems_per_group;
    // const int elems_per_group = at::numel(input_vals) / groups;
    // if (input_vals.scalar_type() == at::ScalarType::Half) {
    //     // launch_quant((int8_t*)output.data_ptr(),
    //     //             (float*)params.data_ptr(),
    //     //             (__half*)input_vals.data_ptr(),
    //     //             groups,
    //     //             elems_per_group,
    //     //             numBits,
    //     //             quantType,
    //     //             at::cuda::getCurrentCUDAStream());
    //     // return {output, params};
    //     throw std::runtime_error("Unsupported input tensor data type.");
    // } else if (input_vals.scalar_type() == at::ScalarType::Float) {
    //     // launch_quant((int8_t*)output.data_ptr(),
    //     //             (float*)params.data_ptr(),
    //     //             (float*)input_vals.data_ptr(),
    //     //             groups,
    //     //             elems_per_group,
    //     //             numBits,
    //     //             quantType,
    //     //             at::cuda::getCurrentCUDAStream());
    //     // return {output, params};
    //     throw std::runtime_error("Unsupported input tensor data type.");
    // } else if (input_vals.scalar_type() == at::ScalarType::BFloat16) {
    //     // launch_sub(
    //     //     (__nv_bfloat16*)output_buffer.data_ptr(),
    //     //     (__nv_bfloat16*)input_vals.data_ptr(),
    //     //     at::numel(input_vals),
    //     //     param_list,
    //     //     dp_param_offset,
    //     //     groups,
    //     //     elems_per_group,
        //     at::cuda::getCurrentCUDAStream());
        launch_sub(
            (__nv_bfloat16*)output_buffer,
            (__nv_bfloat16*)input_vals,
            groups * elems_per_group,
            (__nv_bfloat16*)param_list,
            dp_param_offset,
            groups,
            elems_per_group,
            num_params,
            stream);
        // return output_buffer;
    // } else {
    //     throw std::runtime_error("Unsupported input tensor data type.");
    // }
}