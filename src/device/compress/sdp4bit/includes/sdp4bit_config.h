#ifndef COCCL_SDP4BIT_CONFIG_H_
#define COCCL_SDP4BIT_CONFIG_H_

#include "quantization_utils.h"

struct sdp4bitConfig {
  int groupCount = 2048;
  int quantBits = 8;
  bool hadamard = false;
  quantize::Type quantType = quantize::Type::Symmetric;
  int inQuantBits = 0;
  int outQuantBits = 0;
  int inGroupCount = 0;
  int outGroupCount = 0;
  bool intraAndInter = false;
  int nodes = 1;
  int devicesPerNodes = 4;
  int pipelineSize = 1;
  bool subAdd = false;
};

#endif
