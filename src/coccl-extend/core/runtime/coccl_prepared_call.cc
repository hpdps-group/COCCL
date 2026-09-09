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

unsigned algorithmScopes(const cocclPreparedCall& prepared,
                         cocclAlgorithmKind algorithm) {
  constexpr unsigned useDefault =
      1u << static_cast<unsigned>(cocclCompressionScope::Default);
  constexpr unsigned useIntra =
      1u << static_cast<unsigned>(cocclCompressionScope::Intra);
  constexpr unsigned useInter =
      1u << static_cast<unsigned>(cocclCompressionScope::Inter);
  switch (algorithm) {
    case cocclAlgorithmAllGatherTwoShot:
      return useInter;
    case cocclAlgorithmReduceScatterTwoShot:
      return useIntra | useInter;
    case cocclAlgorithmAllReduceTripleShot:
      return useDefault | useIntra | useInter;
    default:
      return 1u << static_cast<unsigned>(flatScope(prepared.info));
  }
}

}  // namespace

bool cocclPreparedAlgorithmHasCompression(
    const cocclPreparedCall* prepared, cocclAlgorithmKind algorithm) {
  const unsigned scopes = algorithmScopes(*prepared, algorithm);
  for (size_t i = 0; i < prepared->compressors.handles.size(); ++i) {
    if ((scopes & (1u << i)) && prepared->compressors.handles[i] != nullptr) {
      return true;
    }
  }
  return false;
}

bool cocclPreparedAlgorithmSupported(
    const cocclPreparedCall* prepared, cocclAlgorithmKind algorithm) {
  const unsigned scopes = algorithmScopes(*prepared, algorithm);
  bool enabled = false;
  for (size_t i = 0; i < prepared->compressors.handles.size(); ++i) {
    if ((scopes & (1u << i)) && prepared->compressors.handles[i] != nullptr) {
      if (!prepared->compressors.datatypeSupported[i]) return false;
      enabled = true;
    }
  }
  return enabled;
}
