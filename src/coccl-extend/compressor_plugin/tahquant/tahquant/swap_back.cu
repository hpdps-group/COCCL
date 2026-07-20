
#include "quantization.h"
#include <cmath>
namespace cg = cooperative_groups;
template <typename T>
__global__ void swap_back_kernel(T* data, int elems_per_group, const int64_t* max_indices_in, const bool* flag) {
    int64_t group_id = blockIdx.x;
    int64_t tid = threadIdx.x;

    // 只需要每个块的第一个线程来执行恢复操作
    if (tid == 0) {
        if(flag[group_id] == true){
            int64_t group_size = elems_per_group;
            int64_t group_start_index = group_id * group_size;

            // 读取之前记录的该group最大值的索引
            int64_t max_index = max_indices_in[group_id];

            // 再次执行交换操作即可恢复
            if (max_index != -1 && max_index != group_start_index) {
                T temp = data[group_start_index];
                data[group_start_index] = data[max_index];
                data[max_index] = temp;
            }
        }
    }
}

template <typename T>
void launch_swap_back_experimental(T* data_in,
                                    int groups,
                                    int elems_per_group,
                                    int64_t* max_indexs,
                                    bool* flag,
                                    cudaStream_t stream)
{
    // size_t total_elems = groups * elems_per_group;
    dim3 grid(groups);
    int thread_num = 1;
    dim3 block(thread_num);
    swap_back_kernel<<<grid, block, 0, stream>>> (data_in, elems_per_group, max_indexs, flag);
    // printf("sdasdasdsadsadasdsa\n");
    // CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}


template void launch_swap_back_experimental(__half* data_in,
                                             int groups,
                                             int elems_per_group,
                                             int64_t* max_indexs,
                                             bool* flag,
                                             cudaStream_t stream);

template void launch_swap_back_experimental(float* data_in,
                                             int groups,
                                             int elems_per_group,
                                             int64_t* max_indexs,
                                             bool* flag,
                                             cudaStream_t stream);

#ifdef BF16_AVAILABLE
template void launch_swap_back_experimental(__nv_bfloat16* data_in,
                                             int groups,
                                             int elems_per_group,
                                             int64_t* max_indexs,
                                             bool* flag,
                                             cudaStream_t stream);
#endif