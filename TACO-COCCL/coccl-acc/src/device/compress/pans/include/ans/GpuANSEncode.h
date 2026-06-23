/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef MULTIBYTE_ANS_INCLUDE_ANS_GPUANSENCODE_H
#define MULTIBYTE_ANS_INCLUDE_ANS_GPUANSENCODE_H

#pragma once

#include "BatchPrefixSum.h"
#include "GpuANSCodec.h"
#include "GpuANSStatistics.h"
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

namespace multibyte_ans {

template <int ProbBits>
__device__ __forceinline__ uint32_t encodeOne(
    ANSStateT& state,
    ANSDecodedT sym,
    uint32_t outOffset,
    ANSEncodedT* __restrict__ outWords,
    const uint4* __restrict__ table) {
  auto lookup = table[sym];

  uint32_t pdf = lookup.x;
  uint32_t cdf = lookup.y;
  uint32_t div_m1 = lookup.z;
  uint32_t div_shift = lookup.w;

  constexpr ANSStateT kStateCheckMul = 1 << (kANSStateBits - ProbBits);

  ANSStateT maxStateCheck = pdf * kStateCheckMul;
  bool write = state >= maxStateCheck;

  auto vote = __ballot_sync(0xffffffff, write);
  auto prefix = __popc(vote & getLaneMaskLt());

  // Some lanes wish to write out their data
  if (write) {
    outWords[outOffset + prefix] = state & kANSEncodedMask;
    state >>= kANSEncodedBits;
  }

  uint32_t t = __umulhi(state, div_m1);
  //__umulhi 通常是一个内联汇编函数或者内置函数，
  //用于计算两个无符号整数相乘的结果，并且只返回乘积的高半部分（即更显著的位）。
  //这种操作在某些低级编程或者性能敏感的代码中很有用，
  //因为它可以避免处理整个乘积，从而节省空间和时间。
  
  // We prevent addition overflow here by restricting `state` to < 2^31
  // (kANSStateBits)
  uint32_t div = (t + state) >> div_shift;
  //div = (__umulhi(state, div_m1) + state) >> div_shift 
  //= state / div然后向下取整
  auto mod = state - (div * pdf);

  // calculating ((state / pdf) << ProbBits) + (state % pdf) + cdf
  constexpr uint32_t kProbBitsMul = 1 << ProbBits;
  state = div * kProbBitsMul + mod + cdf;

  // how many values we actually write to the compressed output
  return __popc(vote);
}

template <int ProbBits>
__device__ __forceinline__ uint32_t encodeOnePartial(
    // true for the lanes in the warp for which data read is valid
    bool valid,
    ANSStateT& state,
    ANSDecodedT sym,
    uint32_t outOffset,
    ANSEncodedT* __restrict__ outWords,
    const uint4* __restrict__ table) {
  if(!valid) return 0;
  auto lookup = table[sym];

  uint32_t pdf = lookup.x;
  uint32_t cdf = lookup.y;
  uint32_t div_m1 = lookup.z;
  uint32_t div_shift = lookup.w;

  constexpr ANSStateT kStateCheckMul = 1 << (kANSStateBits - ProbBits);

  ANSStateT maxStateCheck = pdf * kStateCheckMul;
  bool write = (state >= maxStateCheck);

  auto vote = __ballot_sync(0xffffffff, write);
  auto prefix = __popc(vote & getLaneMaskLt());

  // Some lanes wish to write out their data
  if (write) {
    outWords[outOffset + prefix] = state & kANSEncodedMask;
    state >>= kANSEncodedBits;
  }

  uint32_t t = __umulhi(state, div_m1);
  //__umulhi 通常是一个内联汇编函数或者内置函数，
  //用于计算两个无符号整数相乘的结果，并且只返回乘积的高半部分（即更显著的位）。
  //这种操作在某些低级编程或者性能敏感的代码中很有用，
  //因为它可以避免处理整个乘积，从而节省空间和时间。
  
  // We prevent addition overflow here by restricting `state` to < 2^31
  // (kANSStateBits)
  uint32_t div = (t + state) >> div_shift;
  //div = (__umulhi(state, div_m1) + state) >> div_shift 
  //= state / div然后向下取整
  auto mod = state - (div * pdf);

  // calculating ((state / pdf) << ProbBits) + (state % pdf) + cdf
  constexpr uint32_t kProbBitsMul = 1 << ProbBits;
  state = div * kProbBitsMul + mod + cdf;

  // how many values we actually write to the compressed output
  return __popc(vote);
}

template <int ProbBits, int BlockSize>
__global__ void ansEncodeBatch(
    uint8_t* in_dev,
    int inSize_dev,
    uint32_t maxNumCompressedBlocks,
    uint32_t uncoalescedBlockStride,
    uint8_t* compressedBlocks_dev,
    uint32_t* compressedWords_dev,
    const uint4* table_dev) {
  uint32_t numBlocks = (inSize_dev + BlockSize - 1) / BlockSize;
  // grid-wide warp id
  int tid = threadIdx.x;
  int grim_warp_numid =//这个grim_warp_numid指的是一个grim中的全局warpid
      __shfl_sync(0xffffffff, (blockIdx.x * blockDim.x + tid) / kWarpSize, 0);
  int laneId = getLaneId();

  __shared__ uint4 smemLookup[kNumSymbols];

  // we always have at least 256 threads
  if (tid < kNumSymbols) {
    smemLookup[tid] = table_dev[tid];
  }
  __syncthreads();

  uint32_t start = grim_warp_numid * BlockSize;
  if(start >= inSize_dev){
    return;
  }

  uint32_t blockSize =  min(start + BlockSize, inSize_dev) - start;

  // Either the warp is an excess one, or the last block is not a full block and
  // needs to be processed using the partial 
  if (grim_warp_numid >= numBlocks)
    return;

  auto inBlock = in_dev + start;
  auto outBlock = (ANSWarpState*)(compressedBlocks_dev
            + grim_warp_numid * uncoalescedBlockStride);

  // all input blocks must meet alignment requirements
  assert(isPointerAligned(inBlock, kANSRequiredAlignment));

  ANSEncodedT* outWords = (ANSEncodedT*)(outBlock + 1);


  ANSStateT state = kANSStartState;

  uint32_t inOffset = laneId;
  uint32_t outOffset = 0;

  constexpr int kUnroll = 8;

  uint32_t limit = roundDown(blockSize, kWarpSize * kUnroll);

  {
    for (; inOffset < limit; inOffset += kWarpSize * kUnroll) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        outOffset +=
            encodeOne<ProbBits>(state, inBlock[inOffset + j * kWarpSize], outOffset, outWords, smemLookup);
      }
    }
  }

  
  if (limit != blockSize) {
    limit = roundDown(blockSize, kWarpSize);

    for (; inOffset < limit; inOffset += kWarpSize) {
      outOffset +=
          encodeOne<ProbBits>(state, inBlock[inOffset], outOffset, outWords, smemLookup);
    }
    //valid = false;
    if (limit != blockSize) {
      bool valid = inOffset < blockSize;
      ANSDecodedT sym = valid ? inBlock[inOffset] : ANSDecodedT(0);
      outOffset += encodeOnePartial<ProbBits>(
          valid, state, sym, outOffset, outWords, smemLookup);
    }
  }
  // Write final state at the beginning (aligned addresses)
  outBlock->warpState[laneId] = state;
  //auto outWord = outOffset;

  if (laneId == 0) {
    //assert(outWord <= getRawCompBlockMaxSize(blockSize) / sizeof(ANSEncodedT));
    compressedWords_dev[grim_warp_numid] = outOffset;
  }
}

template <typename A, int B>
struct Align {
  typedef uint32_t argument_type;
  typedef uint32_t result_type;

  // __thrust_exec_check_disable__ 
  template <typename T>
  __host__ __device__ uint32_t operator()(T x) const {
    constexpr int kDiv = B / sizeof(A);
    constexpr int kSize = kDiv < 1 ? 1 : kDiv;

    return roundUp(x, T(kSize));
  }
};

template <int Threads>
__global__ void ansEncodeCoalesceBatch(
    const uint8_t* __restrict__ compressedBlocks_dev,
    // uint8_t* in_dev,
    int uncompressedWords,// inSize_dev,
    uint32_t maxNumCompressedBlocks,
    uint32_t uncoalescedBlockStride,
    const uint32_t* __restrict__ compressedWords_dev,
    const uint32_t* __restrict__ compressedWordsPrefix_dev,
    const uint4* __restrict__ table_dev,
    uint32_t config_probBits,
    uint8_t* out_dev,
    uint32_t* outSize_dev) {

  auto numBlocks = divUp(uncompressedWords, kDefaultBlockSize);

  int block = blockIdx.x;
  int tid = threadIdx.x;

  ANSCoalescedHeader* headerOut = (ANSCoalescedHeader*)out_dev;

  // The first block will be responsible for the coalesced header
  if (block == 0) {
    if (tid == 0) {
      uint32_t totalCompressedWords = 0;

      // Could be a header for a zero sized array
      if (numBlocks > 0) {
        totalCompressedWords =
            compressedWordsPrefix_dev[numBlocks - 1] +
            roundUp(
                compressedWords_dev[numBlocks - 1],
                kBlockAlignment / sizeof(ANSEncodedT));
      }

      ANSCoalescedHeader header;
      header.setMagicAndVersion();
      header.setNumBlocks(numBlocks);
      header.setTotalUncompressedWords(uncompressedWords);
      header.setTotalCompressedWords(totalCompressedWords);
      header.setProbBits(config_probBits);
      // if(tid == 0 && blockIdx.x == 0)
      // printf("header.setProbBits(config_probBits): %d\n",header.getProbBits());

      if (outSize_dev) {
        *outSize_dev = header.getTotalCompressedSize();
      }

      *headerOut = header;
    }

    auto probsOut = headerOut->getSymbolProbs();

    // Write out pdf
    #pragma unroll
    for (int i = tid; i < kNumSymbols; i += Threads) {
      probsOut[i] = table_dev[i].x;
    }
  }

  if (block >= numBlocks) {
    return;
  }

  // where our per-warp data lies
  auto uncoalescedBlock = compressedBlocks_dev + 
                      block * uncoalescedBlockStride;

  // Write per-block warp state
  if (tid < kWarpSize) {
    auto warpStateIn = (ANSWarpState*)uncoalescedBlock;

    headerOut->getWarpStates()[block].warpState[tid] =
        warpStateIn->warpState[tid];
  }

  auto blockWordsOut = headerOut->getBlockWords(numBlocks);

  // Write out per-block word length
  for (int i = blockIdx.x * Threads + tid; i < numBlocks;
       i += gridDim.x * Threads) {
    uint32_t lastBlockWords = uncompressedWords % kDefaultBlockSize;
    lastBlockWords = lastBlockWords == 0 ? kDefaultBlockSize : lastBlockWords;

    uint32_t blockWords =
        (i == numBlocks - 1) ? lastBlockWords : kDefaultBlockSize;

    blockWordsOut[i] = uint2{
        (blockWords << 16) | compressedWords_dev[i], compressedWordsPrefix_dev[i]};
  }

  // Number of compressed words in this block
  uint32_t numWords = compressedWords_dev[block];

  using LoadT = uint4;

  uint32_t limitEnd = divUp(numWords, kBlockAlignment / sizeof(ANSEncodedT));

  auto inT = (const LoadT*)(uncoalescedBlock + sizeof(ANSWarpState));
  auto outT =
      (LoadT*)(headerOut->getBlockDataStart(numBlocks) + compressedWordsPrefix_dev[block]);

  for (uint32_t i = tid; i < limitEnd; i += Threads) {
    outT[i] = inT[i];
  }
}

void ansEncode(
    uint32_t maxUncompressedWords,
    uint32_t maxNumCompressedBlocks,
    uint4* table_dev,
    uint32_t* tempHistogram_dev,
    uint32_t uncoalescedBlockStride,
    uint8_t* compressedBlocks_dev,
    uint32_t* compressedWords_dev,
    uint32_t* compressedWordsPrefix_dev,
    uint32_t sizeRequired,
    uint8_t* tempPrefixSum_dev,
    int precision,
    uint8_t* in,
    uint32_t inSize,
    uint8_t* out,
    uint32_t* outSize,
    cudaStream_t stream) {

  // uint32_t maxUncompressedWords = inSize / sizeof(ANSDecodedT);
  // uint32_t maxNumCompressedBlocks =
  //     (maxUncompressedWords + kDefaultBlockSize - 1) / kDefaultBlockSize;//一个batch的数据以kDefaultBlockSize作为基准划分数据，形成多个数据块

  // uint4* table_dev;
  // CUDA_VERIFY(cudaMalloc(&table_dev, sizeof(uint4) * kNumSymbols));

  // uint32_t* tempHistogram_dev;
  // CUDA_VERIFY(cudaMalloc(&tempHistogram_dev, sizeof(uint32_t) * kNumSymbols));
// {
// auto start = std::chrono::high_resolution_clock::now();   
  ansHistogramBatch(
      in,
      inSize,
      tempHistogram_dev, 
      stream);
// cudaStreamSynchronize(stream);
// auto end = std::chrono::high_resolution_clock::now();
// double time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;
// printf("histgram kernel: %f\n", time);
// }
// {
// auto start = std::chrono::high_resolution_clock::now();  
  ansCalcWeights(
      precision,
      inSize,
      tempHistogram_dev,
      table_dev,
      stream);
// cudaStreamSynchronize(stream);
// auto end = std::chrono::high_resolution_clock::now();
// double time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;
// printf("weight kernel: %f\n", time);
// }

// auto start = std::chrono::high_resolution_clock::now();  
  // uint32_t uncoalescedBlockStride =
  //     getMaxBlockSizeUnCoalesced(kDefaultBlockSize);

  // uint8_t* compressedBlocks_dev;
  // CUDA_VERIFY(cudaMalloc(&compressedBlocks_dev, sizeof(uint8_t) * maxNumCompressedBlocks * uncoalescedBlockStride));

  // uint32_t* compressedWords_dev;
  // CUDA_VERIFY(cudaMalloc(&compressedWords_dev, sizeof(uint32_t) * maxNumCompressedBlocks));

  // uint32_t* compressedWordsPrefix_dev;
  // CUDA_VERIFY(cudaMalloc(&compressedWordsPrefix_dev, sizeof(uint32_t) * maxNumCompressedBlocks));
// cudaStreamSynchronize(stream);
// auto end = std::chrono::high_resolution_clock::now();
// double time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;
// printf("malloc kernel: %f\n", time);

  if (maxNumCompressedBlocks > 0) {
    constexpr int kThreads = 256;//一个block256线程，4个warp

    auto gridFull = dim3(
        ((int)maxNumCompressedBlocks + (kThreads / kWarpSize)) / (kThreads / kWarpSize), 1);
// auto start = std::chrono::high_resolution_clock::now();   
#define RUN_ENCODE(BITS)                                       \
  do {                                                         \
    ansEncodeBatch<BITS, kDefaultBlockSize>    \
        <<<gridFull, kThreads, 0, stream>>>(                   \
            in,\
            inSize,                                        \
            maxNumCompressedBlocks,                            \
            uncoalescedBlockStride,                            \
            compressedBlocks_dev,                       \
            compressedWords_dev,                        \
            table_dev);                                 \
  } while (false)

    switch (precision) {
      case 9:
        RUN_ENCODE(9);
        break;
      case 10:
        RUN_ENCODE(10);
        break;
      case 11:
        RUN_ENCODE(11);
        break;
      default:
        std::cout<< "unhandled pdf precision " << precision << std::endl;
    }

#undef RUN_ENCODE
// cudaStreamSynchronize(stream);
// auto end = std::chrono::high_resolution_clock::now();
// double time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;
// printf("encoder kernel: %f\n", time);
  }
  if (maxNumCompressedBlocks > 0) {
// {
// auto start = std::chrono::high_resolution_clock::now(); 
    // auto sizeRequired =
    //     getBatchExclusivePrefixSumTempSize(
    //       maxNumCompressedBlocks);
    // uint8_t* tempPrefixSum_dev = nullptr;
    // CUDA_VERIFY(cudaMalloc(&tempPrefixSum_dev, sizeof(uint8_t) * sizeRequired));
    batchExclusivePrefixSum<uint32_t, Align<ANSEncodedT, kBlockAlignment>>(
        compressedWords_dev,
        compressedWordsPrefix_dev,
        tempPrefixSum_dev,
        maxNumCompressedBlocks,
        Align<ANSEncodedT, kBlockAlignment>(),
        stream);
// cudaStreamSynchronize(stream);
// auto end = std::chrono::high_resolution_clock::now();
// double time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;
// printf("prefix kernel: %f\n", time);
//   }
  }
  
  {
// auto start = std::chrono::high_resolution_clock::now(); 
    constexpr int kThreads = 64;
    auto grid = dim3(std::max(maxNumCompressedBlocks, 1U), 1);

    ansEncodeCoalesceBatch< 
    kThreads>
        <<<grid, kThreads, 0, stream>>>(
            compressedBlocks_dev,
            inSize,
            maxNumCompressedBlocks,
            uncoalescedBlockStride,
            compressedWords_dev,
            compressedWordsPrefix_dev,
            table_dev,
            precision,
            out,
            outSize);
// cudaStreamSynchronize(stream);
// auto end = std::chrono::high_resolution_clock::now();
// double time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;
// printf("Coalesce kernel: %f\n\n", time);
  }

  // CUDA_TEST_ERROR();
}

} // namespace 

#undef RUN_ENCODE_ALL

#endif
