/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "ans/GpuANSEncode.h"
#include "ans/GpuANSDecode.h"
#include "ans/GpuANSCodec.h"
#include <cmath>
#include <memory>
#include <vector>

namespace multibyte_ans {

void ansEncodeBatch(
    uint32_t maxNumcompressedWords,
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
        
    ansEncode(
        maxNumcompressedWords,
        maxNumCompressedBlocks,
        table_dev,
        tempHistogram_dev,
        uncoalescedBlockStride,
        compressedBlocks_dev,
        compressedWords_dev,
        compressedWordsPrefix_dev,
        sizeRequired,
        tempPrefixSum_dev,
        precision,
        in,
        inSize,
        out,
        outSize,
        stream);
}

void ansDecodeBatch(
    int precision,
    uint8_t* in,
    uint8_t* out,
    cudaStream_t stream) {

    ansDecode(
        precision,
        in,
        out,
        stream);
}
}