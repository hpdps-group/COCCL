#include "core/pipeline/coccl_pipeline.h"
#include "core/runtime/coccl_comm.h"
#include "core/runtime/coccl_prepared_call.h"
#include "checks.h"
#include "comm.h"
#include "core.h"

NCCL_API(ncclResult_t, cocclAllToAllComp, const void* sendbuff,
  void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm,
  cudaStream_t stream);
ncclResult_t cocclAllToAllComp(const void* sendbuff, void* recvbuff,
  size_t sendcount, ncclDataType_t datatype, ncclComm_t comm,
  cudaStream_t stream) {
  cocclCompressorHandle compressor;
  NCCLCHECK(cocclCommGetCompressor(
      comm, cocclDefaultPolicy(cocclOperation::AllToAll), &compressor));
  cocclPreparedCall prepared;
  prepared.info.sendbuff = sendbuff;
  prepared.info.recvbuff = recvbuff;
  prepared.info.count = sendcount;
  prepared.info.datatype = datatype;
  prepared.info.func = ncclNumFuncs;
  prepared.info.operation = cocclOperation::AllToAll;
  prepared.info.comm = comm;
  prepared.info.stream = stream;
  prepared.descriptor =
      cocclGetOperationDescriptor(cocclOperation::AllToAll);
  prepared.policy = cocclDefaultPolicy(cocclOperation::AllToAll);
  prepared.compressor = compressor;
  return cocclExecutePreparedCall(&prepared);
}

ncclResult_t cocclExecuteAllToAll(const cocclPreparedCall* prepared) {
  const cocclInfo& args = prepared->info;
  const cocclPipelineStage stages[] = {
    cocclPipelineCompress(),
    cocclPipelineAllToAll(args.comm),
    cocclPipelineDecompress(),
  };
  const cocclPipelineSpec spec = {
    "alltoall", args.sendbuff, args.recvbuff, args.count,
    (size_t)args.comm->nRanks, args.datatype, args.comm,
    prepared->compressor, args.stream, stages,
    (int)(sizeof(stages) / sizeof(stages[0])),
    cocclPipelineInPlaceNone
  };
  return cocclRunPipeline(&spec);
}
