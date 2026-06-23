
#include "quantization.h"
#include <cmath>

namespace cg = cooperative_groups;

template <typename T>
__device__ bool getmax(T d1, T d2){
    return fabsf(d1) > fabsf(d2);
    // return true;
}

template<>
__device__ bool getmax<float>(float d1, float d2){
    return fabsf(d1) > fabsf(d2);
}

template<>
__device__ bool getmax<__half>(__half d1, __half d2){
    return fabsf(__half2float(d1)) > fabsf(__half2float(d2));
}

template<>
__device__ bool getmax<__nv_bfloat16>(__nv_bfloat16 d1, __nv_bfloat16 d2){
    return fabsf(__bfloat162float(d1)) > fabsf(__bfloat162float(d2));
}

template <typename T>
__device__ void maxprint(T max1, T max2){
    // printf("max1 %f max2 %f scale %f\n", max1, max2, max1/max2);
    // return true;
}

template <>
__device__ void maxprint<float>(float max1, float max2){
    // if((max1)/(max2) >= 2.0)
    // printf("max1 %f max2 %f scale %f\n", max1, max2, max1/max2);
    // return true;
}

template <>
__device__ void maxprint<__half>(__half max1, __half max2){
    // if(__half2float(max1)/__half2float(max2) >= 2.0)
    // printf("max1 %f max2 %f scale %f\n", __half2float(max1), __half2float(max2), __half2float(max1)/__half2float(max2));
    // return true;
}

template <>
__device__ void maxprint<__nv_bfloat16>(__nv_bfloat16 max1, __nv_bfloat16 max2){
    static uint64_t cnt1 =0 ,cnt2 =0 ;
    // cnt1++;
    // if(__bfloat162float(max1)/__bfloat162float(max2) >= 2.0){
    //     // cnt2 ++;
    //     printf("max1_%f_max2_%f_scale_%f_cnt1_%llu_cnt2_%llu\n", __bfloat162float(max1), __bfloat162float(max2), __bfloat162float(max1)/__bfloat162float(max2), cnt1, cnt2);
    // }
    // return true;
}



template <typename T>
__device__ bool getswap(T d1, T d2){
    return fabsf(d2) * 2 < fabsf(d1);
    // return true;
}

template<>
__device__ bool getswap<float>(float d1, float d2){
    return fabsf(d2) * 2 < fabsf(d1);
}

template<>
__device__ bool getswap<__half>(__half d1, __half d2){
    return fabsf(__half2float(d2)) * 2 < fabsf(__half2float(d1));
}

template<>
__device__ bool getswap<__nv_bfloat16>(__nv_bfloat16 d1, __nv_bfloat16 d2){
    return fabsf(__bfloat162float(d2)) * 2 < fabsf(__bfloat162float(d1));
}


template <typename T>
__global__ void pivot_swap_kernel(T* data, int elems_per_group, int num_groups, int64_t* max_indexs, bool* flag){
    // cg::grid_group grid = cg::this_grid();
    
    extern __shared__ char ss_cache[];
    const int64_t group_id = blockIdx.x;
    const int64_t group_size = elems_per_group;
    const int64_t group_start_index = group_id * group_size;
    struct ValueIndex {
        T max1;
        T max2;
        int64_t index;
    };
    ValueIndex* s_cache = (ValueIndex *)ss_cache;
    const int64_t tid = threadIdx.x;
    const int64_t stride = blockDim.x;
    

    ValueIndex thread_max = {0, 0, -1};


    for (int64_t i = group_start_index + tid; i < group_start_index + group_size; i += stride){
        // if (fabsf(data[i]) > fabsf(thread_max.value)){
        //     thread_max.value = data[i];
        //     thread_max.index = i;
        // }
        if (getmax(data[i], thread_max.max1)){
            thread_max.max2 = thread_max.max1;
            thread_max.max1 = data[i];
            thread_max.index = i;
        }else if (getmax(data[i], thread_max.max2)){
            thread_max.max2 = data[i];
        }
    }

    s_cache[tid] = thread_max;

    __syncthreads();

    for (int64_t s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s){
            // if(fabsf(s_cache[tid + s].value) > fabsf(s_cache[tid].value)){
            //     s_cache[tid] = s_cache[tid + s];
            // }
            // if(getmax(s_cache[tid + s].value, s_cache[tid].value)){
            //     s_cache[tid] = s_cache[tid + s];
            // }
            if(getmax(s_cache[tid + s].max1, s_cache[tid].max1)){
                if(getmax(s_cache[tid + s].max2, s_cache[tid].max1)){
                    s_cache[tid].max2 = s_cache[tid + s].max2;
                } else {
                    s_cache[tid].max2 = s_cache[tid].max1;
                }
                s_cache[tid].max1 = s_cache[tid + s].max1;
                s_cache[tid].index = s_cache[tid + s].index;
            }else if(getmax(s_cache[tid + s].max1, s_cache[tid].max2)){
                s_cache[tid].max2 = s_cache[tid + s].max1;
            }
        }
        __syncthreads();
    }

    if (tid == 0){
        // maxprint(s_cache[0].max1, s_cache[0].max2);
        flag[group_id] = 0;
        if(getswap(s_cache[0].max1, s_cache[0].max2)){
            const int64_t group_max_index = s_cache[0].index;
            max_indexs[group_id] = group_max_index;
            flag[group_id] = 1;
            // printf("swap_group_%lld\n", group_id);
            if (group_max_index != -1 && group_max_index != group_start_index){
                T first_element_val = data[group_start_index];
                data[group_start_index] = s_cache[0].max1;
                data[group_max_index] = first_element_val;
                // printf("s_cache[0].value %f\n",s_cache[0].value);
            }
        }
        
    }
}

template <typename T>
void launch_pivot_swap_experimental(T* data_in,
                                    int groups,
                                    int elems_per_group,
                                    int64_t* max_indexs,
                                    bool* flag,
                                    cudaStream_t stream)
{
    // size_t total_elems = groups * elems_per_group;
    dim3 grid(groups);
    int thread_num = elems_per_group < 256 ? elems_per_group : 256;
    dim3 block(thread_num);
    struct ValueIndex {
        T max1;
        T max2;
        int64_t index;
    };
    size_t shared_mem_size = thread_num * (sizeof(ValueIndex));
    pivot_swap_kernel<<<grid, block, shared_mem_size, stream>>> (data_in, elems_per_group, groups, max_indexs, flag);
    // printf("sdasdasdsadsadasdsa\n");
    // CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}


template void launch_pivot_swap_experimental(__half* data_in,
                                             int groups,
                                             int elems_per_group,
                                             int64_t* max_indexs,
                                             bool* flag,
                                             cudaStream_t stream);

template void launch_pivot_swap_experimental(float* data_in,
                                             int groups,
                                             int elems_per_group,
                                             int64_t* max_indexs,
                                             bool* flag,
                                             cudaStream_t stream);

#ifdef BF16_AVAILABLE
template void launch_pivot_swap_experimental(__nv_bfloat16* data_in,
                                             int groups,
                                             int elems_per_group,
                                             int64_t* max_indexs,
                                             bool* flag,
                                             cudaStream_t stream);
#endif