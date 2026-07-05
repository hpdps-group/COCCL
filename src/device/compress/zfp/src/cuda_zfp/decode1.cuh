#ifndef CUZFP_DECODE1_CUH
#define CUZFP_DECODE1_CUH

#include "shared.h"
#include "decode.cuh"
#include "type_info.cuh"

namespace cuZFP {


template<typename Scalar> 
__device__ __host__ inline 
void scatter_partial1(const Scalar* q, Scalar* p, int nx, int sx, zfpEXDType_t T1)
{
  uint x;
  for (x = 0; x < 4; x++)
    if (x < nx) {
      if(T1 == zfpEXDType_t::zfpBfloat16){
        reinterpret_cast<__nv_bfloat16*>(p)[x * sx] = __float2bfloat16(q[x]);
      }
      else if(T1 == zfpEXDType_t::zfpFloat16 || T1 == zfpEXDType_t::zfpHalf)
        reinterpret_cast<__half*>(p)[x * sx] = __float2half(q[x]);
      else {
        p[x * sx] = q[x];
      }
    }
}

template<typename Scalar> 
__device__ __host__ inline 
void scatter1(const Scalar* q, Scalar* p, int sx, zfpEXDType_t T1)
{
  uint x;
  if(T1 == zfpEXDType_t::zfpBfloat16){
    __nv_bfloat16* t_p = reinterpret_cast<__nv_bfloat16*>(p);
    for (x = 0; x < 4; x++, t_p += sx)
      *t_p = __float2bfloat16(*q++);
  }
  else if(T1 == zfpEXDType_t::zfpFloat16 || T1 == zfpEXDType_t::zfpHalf){
    __half* t_p = reinterpret_cast<__half*>(p);
    for (x = 0; x < 4; x++, t_p += sx)
      *t_p = __float2half(*q++);
  }
  else {
    for (x = 0; x < 4; x++, p += sx)
      *p = *q++;
  }
}

template<class Scalar>
__global__
void
cudaDecode1(Word *blocks,
            Scalar *out,
            const uint dim,
            const int stride,
            const uint padded_dim,
            const uint total_blocks,
            uint maxbits,
            zfpEXDType_t T1)
{
  typedef unsigned long long int ull;
  typedef long long int ll;
  typedef typename zfp_traits<Scalar>::UInt UInt;
  typedef typename zfp_traits<Scalar>::Int Int;

  const int intprec = get_precision<Scalar>();

  const ull blockId = blockIdx.x +
                      blockIdx.y * gridDim.x +
                      gridDim.x  * gridDim.y * blockIdx.z;

  // each thread gets a block so the block index is 
  // the global thread index
  const ull block_idx = blockId * blockDim.x + threadIdx.x;

  if(block_idx >= total_blocks) return;
  BlockReader<4> reader(blocks, maxbits, block_idx, total_blocks);
  Scalar result[4] = {0,0,0,0};

  zfp_decode(reader, result, maxbits);

  uint block;
  block = block_idx * 4ull; 
  const ll offset = (ll)block * stride; 
  
  bool partial = false;
  if(block + 4 > dim) partial = true;
  if(partial)
  {
    const uint nx = 4u - (padded_dim - dim);
    if(T1 == zfpEXDType_t::zfpBfloat16 || T1 == zfpEXDType_t::zfpFloat16 ||  T1 == zfpEXDType_t::zfpHalf){
      __nv_bfloat16* t_out = reinterpret_cast<__nv_bfloat16*>(out);
      scatter_partial1(result, reinterpret_cast<Scalar*>(t_out + offset), nx, stride, T1);
    } else {
      scatter_partial1(result, out + offset, nx, stride, T1);
    }
  }
  else
  {
    if(T1 == zfpEXDType_t::zfpBfloat16 ||  T1 == zfpEXDType_t::zfpFloat16 ||  T1 == zfpEXDType_t::zfpHalf){
      __nv_bfloat16* t_out = reinterpret_cast<__nv_bfloat16*>(out);
      scatter1(result, reinterpret_cast<Scalar*>(t_out + offset), stride, T1);
    } else {
      scatter1(result, out + offset, stride, T1);
    }
  }
}

template<class Scalar>
size_t decode1launch(uint dim, 
                     int stride,
                     Word *stream,
                     Scalar *d_data,
                     uint maxbits,
                     zfpEXDType_t T1, 
                     cudaStream_t exec_stream)
{
  const int cuda_block_size = 128;

  uint zfp_pad(dim); 
  if(zfp_pad % 4 != 0) zfp_pad += 4 - dim % 4;

  uint zfp_blocks = (zfp_pad) / 4; 

  if(dim % 4 != 0)  zfp_blocks = (dim + (4 - dim % 4)) / 4;

  int block_pad = 0;
  if(zfp_blocks % cuda_block_size != 0) 
  {
    block_pad = cuda_block_size - zfp_blocks % cuda_block_size; 
  }

  size_t total_blocks = block_pad + zfp_blocks;
  size_t stream_bytes = calc_device_mem1d(zfp_pad, maxbits);

  dim3 block_size = dim3(cuda_block_size, 1, 1);
  dim3 grid_size = calculate_grid_size(total_blocks, cuda_block_size);

#ifdef CUDA_ZFP_RATE_PRINT
  // setup some timing code
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, exec_stream);
#endif
// for(int i =0; i < 100; i++){
  // cudaMemset(stream, 0, stream_bytes);
  cudaDecode1<Scalar> << < grid_size, block_size, 0, exec_stream>>>
    (stream,
		 d_data,
     dim,
     stride,
     zfp_pad,
     zfp_blocks, // total blocks to decode
     maxbits,
     T1);
  // }


#ifdef CUDA_ZFP_RATE_PRINT
  cudaEventRecord(stop, exec_stream);
  cudaEventSynchronize(stop);
  cudaStreamWaitEvent(0, stop, 0);
	cudaStreamSynchronize(0);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  float seconds = milliseconds / 1000.f;
  float rate = (float(dim) * sizeof(Scalar) ) / seconds;
  rate /= 1024.f;
  rate /= 1024.f;
  rate /= 1024.f;
  printf("Decode elapsed time: %.5f (s)\n", seconds);
  printf("# decode1 rate: %.2f (GB / sec) %d\n", rate, maxbits);
#endif
  return stream_bytes;
}

template<class Scalar>
size_t decode1(int dim, 
               int stride,
               Word *stream,
               Scalar *d_data,
               uint maxbits,
               zfpEXDType_t T1, 
               cudaStream_t exec_stream)
{
	return decode1launch<Scalar>(dim, stride, stream, d_data, maxbits, T1, exec_stream);
}

} // namespace cuZFP

#endif
