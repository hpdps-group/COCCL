#include "core/runtime/coccl_prepared_call.h"

#include "comm.h"

namespace {

cocclCompressionScope flatScope(const cocclInfo& info) {
  if (info.operation == cocclOperation::SendRecv) {
    return info.comm->rankToNode[info.peer] == info.comm->node
        ? cocclCompressionScope::Intra
        : cocclCompressionScope::Inter;
  }
  return info.comm->nNodes == 1
      ? cocclCompressionScope::Intra
      : cocclCompressionScope::Default;
}

void algorithmScopes(const cocclPreparedCall& prepared,
                     cocclAlgorithmKind algorithm,
                     bool* useDefault, bool* useIntra, bool* useInter) {
  *useDefault = false;
  *useIntra = false;
  *useInter = false;
  if (algorithm == cocclAlgorithmReduceScatterTwoShot) {
    *useIntra = true;
    *useInter = true;
    return;
  }
  if (algorithm == cocclAlgorithmAllReduceTripleShot) {
    *useDefault = true;
    *useIntra = true;
    *useInter = true;
    return;
  }
  switch (flatScope(prepared.info)) {
    case cocclCompressionScope::Default: *useDefault = true; break;
    case cocclCompressionScope::Intra: *useIntra = true; break;
    case cocclCompressionScope::Inter: *useInter = true; break;
    case cocclCompressionScope::Count: break;
  }
}

bool hasCompression(const cocclPreparedCall& prepared, bool useDefault,
                    bool useIntra, bool useInter) {
  return (useDefault && prepared.compressors.get(
                            cocclCompressionScope::Default) != nullptr) ||
      (useIntra && prepared.compressors.get(
                       cocclCompressionScope::Intra) != nullptr) ||
      (useInter && prepared.compressors.get(
                       cocclCompressionScope::Inter) != nullptr);
}

}  // namespace

bool cocclPreparedAlgorithmHasCompression(
    const cocclPreparedCall* prepared, cocclAlgorithmKind algorithm) {
  bool useDefault = false;
  bool useIntra = false;
  bool useInter = false;
  algorithmScopes(
      *prepared, algorithm, &useDefault, &useIntra, &useInter);
  return hasCompression(*prepared, useDefault, useIntra, useInter);
}

bool cocclPreparedAlgorithmSupported(
    const cocclPreparedCall* prepared, cocclAlgorithmKind algorithm) {
  bool useDefault = false;
  bool useIntra = false;
  bool useInter = false;
  algorithmScopes(
      *prepared, algorithm, &useDefault, &useIntra, &useInter);
  if (!hasCompression(*prepared, useDefault, useIntra, useInter)) {
    return false;
  }
  return (!useDefault ||
          prepared->compressors.get(cocclCompressionScope::Default) ==
              nullptr ||
          prepared->compressors.supports(cocclCompressionScope::Default)) &&
      (!useIntra ||
       prepared->compressors.get(cocclCompressionScope::Intra) == nullptr ||
       prepared->compressors.supports(cocclCompressionScope::Intra)) &&
      (!useInter ||
       prepared->compressors.get(cocclCompressionScope::Inter) == nullptr ||
       prepared->compressors.supports(cocclCompressionScope::Inter));
}
