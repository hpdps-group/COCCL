#ifndef COCCL_SIZE_QUERY_STUB_H_
#define COCCL_SIZE_QUERY_STUB_H_

#include "compressor_plugin/detail/coccl_compressor_abi.h"

struct cocclSizeQueryObservation {
  int calls;
  size_t elements;
  size_t chunks;
  ncclDataType_t datatype;
};

void cocclResetSizeQueryStub();
void cocclConfigureSizeQueryStub(
    size_t compressNumerator, size_t compressDenominator,
    size_t drcNumerator, size_t drcDenominator, bool supportsDrc);
void cocclConfigureFramedSizeQueryStub(bool framed);
void cocclConfigureFusedSwizzleStub(bool fusedSwizzle);
const cocclSizeQueryObservation& cocclCompressQueryObservation();
const cocclSizeQueryObservation& cocclDrcQueryObservation();

#endif
