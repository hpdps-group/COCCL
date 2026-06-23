/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef MULTIBYTE_ANS_INCLUDE_ANS_GPUANSDECODE_H
#define MULTIBYTE_ANS_INCLUDE_ANS_GPUANSDECODE_H

#pragma once

#include "GpuANSCodec.h"
#include "utils/DeviceUtils.h"
#include "utils/PtxUtils.h"
#include "utils/StaticUtils.h"
#include <cmath>
#include <cub/block/block_scan.cuh>
#include <memory>
#include <sstream>
#include <vector>

namespace multibyte_ans {

// We are limited to 11 bits of probability resolution
// (worst case, prec = 12, pdf == 2^12, single symbol. 2^12 cannot be
// represented in 12 bits)
inline __device__ uint32_t
packDecodeLookup(uint32_t sym, uint32_t pdf, uint32_t cdf) {
  static_assert(sizeof(ANSDecodedT) == 1, "");
  // [31:20] cdf
  // [19:8] pdf
  // [7:0] symbol
  return (cdf << 20) | (pdf << 8) | sym;
}

inline __device__ void
unpackDecodeLookup(uint32_t v, uint32_t& sym, uint32_t& pdf, uint32_t& cdf) {
  // [31:20] cdf
  // [19:8] pdf
  // [7:0] symbol
  sym = v & 0xffU;
  v >>= 8;
  pdf = v & 0xfffU;
  v >>= 12;
  cdf = v;
}

template <int ProbBits>
__device__ void decodeOneWarp(
    ANSStateT& state,

    uint32_t compressedOffset,

    const ANSEncodedT* __restrict__ in,

    // Shared memory LUTs
    const uint32_t* lookup,

    // Output: number of words read from compressed input
    uint32_t& outNumRead,

    // Output: decoded symbol for this iteration
    ANSDecodedT& outSym) {
  constexpr ANSStateT StateMask = (ANSStateT(1) << ProbBits) - ANSStateT(1);

  auto s_bar = state & StateMask;

  uint32_t sym;
  uint32_t pdf;
  uint32_t sMinusCdf;
  unpackDecodeLookup(lookup[s_bar], sym, pdf, sMinusCdf);

  // We always write a decoded value
  outSym = sym;
  state = pdf * (state >> ProbBits) + ANSStateT(sMinusCdf);

  // We only sometimes read a new encoded value
  bool read = state < kANSMinState;
  auto vote = __ballot_sync(0xffffffff, read);
  // We are reading in the same order as we wrote, except by decrementing from
  // compressedOffset, so we need to count down from the highest lane in the
  // warp
  auto prefix = __popc(vote & getLaneMaskGe());

  if (read) {
    // auto v = in[compressedOffset - prefix];
    auto v = in[-prefix];
    state = (state << kANSEncodedBits) + ANSStateT(v);
  }

  // how many values we actually read from the compressed input
  outNumRead = __popc(vote);
}

template <int ProbBits>
__device__ void decodeOnePartialWarp(
    bool valid,
    ANSStateT& state,

    uint32_t compressedOffset,

    const ANSEncodedT* __restrict__ in,

    // Shared memory LUTs
    const uint32_t* lookup,

    // Output: number of words read from compressed input
    uint32_t& outNumRead,

    // Output: decoded symbol for this iteration (only if valid)
    ANSDecodedT& outSym) {
  constexpr ANSStateT StateMask = (ANSStateT(1) << ProbBits) - ANSStateT(1);

  auto s_bar = state & StateMask;

  uint32_t sym;
  uint32_t pdf;
  uint32_t sMinusCdf;
  unpackDecodeLookup(lookup[s_bar], sym, pdf, sMinusCdf);

  if (valid) {
    outSym = sym;
    state = pdf * (state >> ProbBits) + ANSStateT(sMinusCdf);
  }

  // We only sometimes read a new encoded value
  bool read = valid && (state < kANSMinState);
  auto vote = __ballot_sync(0xffffffff, read);
  // We are reading in the same order as we wrote, except by decrementing from
  // compressedOffset, so we need to count down from the highest lane in the
  // warp
  auto prefix = __popc(vote & getLaneMaskGe());

  if (read) {
    // auto v = in[compressedOffset - prefix];
    auto v = in[-prefix];
    state = (state << kANSEncodedBits) + ANSStateT(v);
  }

  // how many values we actually read from the compressed input
  outNumRead = __popc(vote);
}

template <int ProbBits>
__device__ void ansDecodeWarpBlock(
    int laneId,
    ANSStateT state,
    uint32_t uncompressedWords,
    uint32_t compressedWords,
    const ANSEncodedT* __restrict__ in,
    BatchWriter& writer,
    const uint32_t* __restrict__ table) {
  // The compressed input may not be a whole multiple of a warp.
  // In this case, only the lanes that cover the remainder could have read a
  // value in the input, and thus, only they can write a value in the output.
  // We handle this partial data first.
  uint32_t remainder = uncompressedWords % kWarpSize;

  // A fixed number of uncompressed elements are written each iteration
  int uncompressedOffset = uncompressedWords - remainder;

  // A variable number of compressed elements are read each iteration
  uint32_t compressedOffset = compressedWords;

  in += compressedOffset;

  // Partial warp handling the end of the data
  if (remainder) {
    bool valid = laneId < remainder;

    uint32_t numCompressedRead;
    ANSDecodedT sym;

    decodeOnePartialWarp<ProbBits>(
        valid, state, compressedOffset, in, table, numCompressedRead, sym);

    if (valid) {
      writer.write(uncompressedOffset + laneId, sym);
    }

    // compressedOffset -= numCompressedRead;
    in -= numCompressedRead;
  }

  // Full warp handling
  while (uncompressedOffset > 0) {
    uncompressedOffset -= kWarpSize;

    uint32_t numCompressedRead;
    ANSDecodedT sym;

    decodeOneWarp<ProbBits>(
        state, compressedOffset, in, table, numCompressedRead, sym);

    writer.write(uncompressedOffset + laneId, sym);

    // compressedOffset -= numCompressedRead;
    in -= numCompressedRead;
  }
}

template <
    // typename InProvider,
    // typename OutProvider,
    int Threads,
    int ProbBits,
    int BlockSize>
__global__ __launch_bounds__(128) void ansDecodeKernel(
    //InProvider inProvider,
    void* in,
    uint32_t* table,
    //OutProvider outProvider,
    void* out
    ) {
  int tid = threadIdx.x;
  //auto batch = blockIdx.y;

  // Interpret header as uint4
  auto headerIn = (const ANSCoalescedHeader*)in;
  headerIn->checkMagicAndVersion();

  auto header = *headerIn;
  auto numBlocks = header.getNumBlocks();
  auto totalUncompressedWords = header.getTotalUncompressedWords();

  // Is the data what we expect?
  assert(ProbBits == header.getProbBits());
  
  // Initialize symbol, pdf, cdf tables
  constexpr int kBuckets = 1 << ProbBits;
  __shared__ uint32_t lookup[kBuckets];

  {
    uint4* lookup4 = (uint4*)lookup;
    const uint4* table4 = (const uint4*)table;
    // + batch * (1 << ProbBits));

    static_assert(isEvenDivisor(kBuckets, Threads * 4), "");
    for (int j = 0;
         // loading by uint4 words
         j < kBuckets / (Threads * (sizeof(uint4) / sizeof(uint32_t)));
         ++j) {
      lookup4[j * Threads + tid] = table4[j * Threads + tid];
    }
  }

  __syncthreads();

  //auto writer = outProvider.getWriter(batch);
  auto writer = BatchWriter(out);

  // warp id taking into account warps in the current block
  // do this so the compiler knows it is warp uniform
  int globalWarpId =
      __shfl_sync(0xffffffff, (blockIdx.x * blockDim.x + tid) / kWarpSize, 0);

  int warpsPerGrid = gridDim.x * Threads / kWarpSize;
  int laneId = getLaneId();

  for (int block = globalWarpId; block < numBlocks; block += warpsPerGrid) {
    // Load state
    ANSStateT state = headerIn->getWarpStates()[block].warpState[laneId];

    // Load per-block size data
    auto blockWords = headerIn->getBlockWords(numBlocks)[block];
    uint32_t uncompressedWords = (blockWords.x >> 16);
    uint32_t compressedWords = (blockWords.x & 0xffff);
    uint32_t blockCompressedWordStart = blockWords.y;

    // Get block addresses for encoded/decoded data
    auto blockDataIn =
        headerIn->getBlockDataStart(numBlocks) + blockCompressedWordStart;

    writer.setBlock(block);

    //using Writer = typename BatchWriter;
    if (uncompressedWords == BlockSize) {
      blockDataIn += compressedWords;

      for (int i = BlockSize - kWarpSize + laneId; i >= 0; i -= kWarpSize) {
        ANSDecodedT sym;
        uint32_t numCompressedRead;

        decodeOneWarp<ProbBits>(
            state, compressedWords, blockDataIn, lookup, numCompressedRead, sym);

        blockDataIn -= numCompressedRead;

        writer.write(i, sym);
      }
    } else {
      ansDecodeWarpBlock<ProbBits>(
          laneId,
          state,
          uncompressedWords,
          compressedWords,
          blockDataIn,
          writer,
          lookup);
    }
  }
}

template <
//typename BatchProvider, 
int Threads>
__global__ void ansDecodeTable(
    //BatchProvider inProvider,
    void* in,
    uint32_t probBits,
    uint32_t* __restrict__ table) {
  //int batch = blockIdx.x;
  int tid = threadIdx.x;
  int warpId = tid / kWarpSize;
  int laneId = getLaneId();

  //table += batch * (1 << probBits);
  auto headerIn = (const ANSCoalescedHeader*)in;

  auto header = *headerIn;

  // Is this an expected header?
  //header.checkMagicAndVersion();

  // Is our probability resolution what we expected?
  //assert(int(header.getProbBits()) == int(probBits));

  if (header.getTotalUncompressedWords() == 0) {
    return;
  }

  // Skip to pdf table
  auto probs = headerIn->getSymbolProbs();

  //static_assert(Threads >= kNumSymbols, "");
  uint32_t pdf = tid < kNumSymbols ? probs[tid] : 0;
  uint32_t cdf = 0;

  // Get the CDF from the PDF
  using BlockScan = cub::BlockScan<uint32_t, Threads>;
  __shared__ typename BlockScan::TempStorage tempStorage;

  uint32_t total = 0;
  // FIXME: don't use cub, we can write both the pdf and cdf to smem with a
  // single syncthreads
  BlockScan(tempStorage).ExclusiveSum(pdf, cdf, total);

  // uint32_t totalProb = 1 << probBits;
  // assert(totalProb == total); // should be a power of 2

  // Broadcast the pdf/cdf values
  __shared__ uint2 smemPdfCdf[kNumSymbols];

  if (tid < kNumSymbols) {
    smemPdfCdf[tid] = uint2{pdf, cdf};
  }

  __syncthreads();

  // Build the table for each pdf/cdf bucket
  constexpr int kWarpsPerBlock = Threads / kWarpSize;

  for (int i = warpId; i < kNumSymbols; i += kWarpsPerBlock) {
    auto v = smemPdfCdf[i];

    auto pdf = v.x;
    auto begin = v.y;
    auto end = begin + pdf;

    for (int j = begin + laneId; j < end; j += kWarpSize) {
      table[j] = packDecodeLookup(
          i, // symbol
          pdf, // bucket pdf
          j - begin); // within-bucket cdf
    }
  }
}

void ansDecode(
    int precision,
    uint8_t* in,
    uint8_t* out,
    cudaStream_t stream) {
  // auto table_dev =
  //     res.alloc<Uint32_t>(stream, numInBatch * (1 << config.probBits));
  uint32_t* table_dev;
  CUDA_VERIFY(cudaMalloc(&table_dev, (1 << precision) * sizeof(uint32_t)));

  // Build the rANS decoding table from the compression header
  {
    constexpr int kThreads = 512;
    ansDecodeTable<kThreads><<<1, kThreads, 0, stream>>>(
        in,
        precision, table_dev);
  }

  // Perform decoding
  {
    // FIXME: We have no idea how large the decompression job is, as the
    // relevant information is on the device.
    // Just launch a grid that is sufficiently large enough to saturate the GPU;
    // blocks will exit if there isn't enough work, or will loop if there is
    // more work. We aim for a grid >4x larger than what the device can sustain,
    // to help cover up tail effects and unequal provisioning across the batch
#define RUN_DECODE(BITS)                                           \
  do {                                                             \
    constexpr int kThreads = 128;                                  \
    auto& props = getCurrentDeviceProperties();                    \
    int maxBlocksPerSM = 0;                                        \
    CUDA_VERIFY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(     \
        &maxBlocksPerSM,                                           \
        ansDecodeKernel<                                           \
            kThreads,                                              \
            BITS,                                                  \
            kDefaultBlockSize>,                                    \
        kThreads,                                                  \
        0));                                                       \
                                                                   \
    uint32_t maxGrid = maxBlocksPerSM * props.multiProcessorCount; \
    uint32_t perBatchGrid = maxGrid * 4;                           \
    auto grid = dim3(perBatchGrid, 1);                             \
    ansDecodeKernel<                                               \
        kThreads,                                                  \
        BITS,                                                      \
        kDefaultBlockSize><<<grid, kThreads, 0, stream>>>(         \
        in,                                                        \
        table_dev,                                                 \
        out                                                        \
        );                                                         \
  } while (false)


    switch (precision) {
      case 9:
        RUN_DECODE(9);
        break;
      case 10:
        RUN_DECODE(10);
        break;
      case 11:
        RUN_DECODE(11);
        break;
      default:
        std::cout << "unhandled pdf precision " << precision << std::endl;
    }

#undef RUN_DECODE
  }

  CUDA_TEST_ERROR();
}

} // namespace 

#endif