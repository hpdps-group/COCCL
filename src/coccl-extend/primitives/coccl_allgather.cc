#include "primitives/coccl_primitives_internal.h"

NCCL_API(ncclResult_t, ncclAllGatherCompOverlap, const void* sendbuff,
  void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm,
  cudaStream_t stream);
ncclResult_t ncclAllGatherCompOverlap(const void* sendbuff, void* recvbuff,
  size_t sendcount, ncclDataType_t datatype, ncclComm_t comm,
  cudaStream_t stream) {
  cocclCompressorHandle compressor;
  NCCLCHECK(cocclCommGetCompressor(
      comm, cocclDefaultPolicy(cocclOperation::AllGather), &compressor));
  cocclPreparedCall prepared;
  prepared.info.sendbuff = sendbuff;
  prepared.info.recvbuff = recvbuff;
  prepared.info.count = sendcount;
  prepared.info.datatype = datatype;
  prepared.info.func = ncclFuncAllGather;
  prepared.info.operation = cocclOperation::AllGather;
  prepared.info.comm = comm;
  prepared.info.stream = stream;
  prepared.descriptor =
      cocclGetOperationDescriptor(cocclOperation::AllGather);
  prepared.policy = cocclDefaultPolicy(cocclOperation::AllGather);
  prepared.compressor = compressor;
  return cocclExecutePreparedCall(&prepared);
}

ncclResult_t cocclExecuteAllGather(const cocclPreparedCall* prepared) {
  if (prepared == nullptr || prepared->info.comm == nullptr ||
      !prepared->compressor) {
    return ncclInvalidArgument;
  }
  const cocclInfo& args = prepared->info;
  const cocclPipelineStage stages[] = {
    cocclPipelineCompress(),
    cocclPipelineAllGather(args.comm),
    cocclPipelineDecompress(),
  };
  const cocclPipelineSpec spec = {
    "allgather-overlap", args.sendbuff, args.recvbuff, args.count, 1,
    args.datatype, args.comm, prepared->compressor, args.stream, stages,
    (int)(sizeof(stages) / sizeof(stages[0])),
    cocclPipelineInPlaceInputRankChunk
  };
  return cocclRunPipeline(&spec);
}
