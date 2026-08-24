#ifndef CUEXDTYPE_CUH
#define CUEXDTYPE_CUH

#include <cuda_fp16.h>
#include <cuda_bf16.h>

typedef enum { 
    zfpOrigin = 0,
    zfpFloat16    = 1, zfpHalf       = 1,
   #if defined(__CUDA_BF16_TYPES_EXIST__)
    zfpBfloat16   = 2,
  #endif
} zfpEXDType_t;
  
#endif