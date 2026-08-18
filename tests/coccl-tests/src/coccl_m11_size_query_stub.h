#ifndef COCCL_M11_SIZE_QUERY_STUB_H_
#define COCCL_M11_SIZE_QUERY_STUB_H_

#include "compressor_plugin/detail/coccl_compressor_abi.h"

struct cocclM11SizeQueryObservation {
  int calls;
  size_t elements;
  size_t chunks;
  ncclDataType_t datatype;
};

void cocclM11ResetSizeQueryStub();
void cocclM11ConfigureSizeQueryStub(
    size_t compressNumerator, size_t compressDenominator,
    size_t drcNumerator, size_t drcDenominator, bool supportsDrc);
const cocclM11SizeQueryObservation& cocclM11CompressQueryObservation();
const cocclM11SizeQueryObservation& cocclM11DrcQueryObservation();

#endif
