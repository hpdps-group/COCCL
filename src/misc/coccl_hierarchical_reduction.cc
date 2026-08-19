#include "coccl_hierarchical_reduction.h"

#include "coccl_prepared_call.h"
#include "comm.h"

namespace {

void append(cocclPipelineStage stage, cocclPipelineStage* stages,
            int* count) {
  stages[(*count)++] = stage;
}

}  // namespace

int cocclBuildHierarchicalReduction(
    const cocclPreparedCall* prepared, ncclComm_t intraComm,
    ncclComm_t interComm, void* finalCompressor,
    cocclPipelineStage* stages) {
  void* const intra =
      prepared->compressors.get(cocclCompressionScope::Intra);
  void* const inter =
      prepared->compressors.get(cocclCompressionScope::Inter);
  int count = 0;

  if (intra != nullptr) {
    append(cocclPipelineCompress(intra), stages, &count);
    append(cocclPipelineAllToAll(intraComm), stages, &count);
    if (inter != nullptr) {
      append(cocclPipelineDecompReduceComp(
                 (size_t)intraComm->nRanks, inter),
             stages, &count);
    } else {
      append(cocclPipelineDecompressReduce(
                 (size_t)intraComm->nRanks),
             stages, &count);
    }
  } else {
    append(cocclPipelineReduceScatter(intraComm), stages, &count);
    if (inter != nullptr) {
      append(cocclPipelineCompress(inter), stages, &count);
    }
  }

  if (inter != nullptr) {
    append(cocclPipelineAllToAll(interComm), stages, &count);
    if (finalCompressor != nullptr) {
      append(cocclPipelineDecompReduceComp(
                 (size_t)interComm->nRanks, finalCompressor),
             stages, &count);
    } else {
      append(cocclPipelineDecompressReduce(
                 (size_t)interComm->nRanks),
             stages, &count);
    }
  } else {
    append(cocclPipelineReduceScatter(interComm), stages, &count);
    if (finalCompressor != nullptr) {
      append(cocclPipelineCompress(finalCompressor), stages, &count);
    }
  }

  return count;
}
