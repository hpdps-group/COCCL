#include "core/pipeline/coccl_pipeline.h"

// Compile-only contract for the COCCL-private linear pipeline API. Runtime
// behavior is exercised by the collective-specific overlap performance tests.
void cocclPipelineContract(const void* input, void* output, ncclComm_t comm,
                           ncclComm_t intraComm, ncclComm_t interComm,
                           cudaStream_t stream) {
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(),
      cocclPipelineAllToAll(intraComm),
      cocclPipelineDecompReduceComp(8),
      cocclPipelineAllToAll(interComm),
      cocclPipelineDecompressReduce(2),
      cocclPipelineAllGather(comm),
      cocclPipelineDecompress(),
  };
  cocclPipelineSpec spec = {
      "contract", input, output, 1024, 16, ncclFloat32, comm,
      cocclCompressorHandle{}, stream, stages, 7,
      cocclPipelineInPlaceNone};
  (void)cocclRunPipeline(&spec);
}
