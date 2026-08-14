#include "primitives/coccl_primitives_internal.h"

namespace {

ncclResult_t cocclRunReduceScatterOneShot(
    const cocclPreparedCall* prepared) {
  const cocclInfo& args = prepared->info;
  ncclComm_t comm = args.comm;
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(),
      cocclPipelineAllToAll(comm),
      cocclPipelineDecompressReduce((size_t)comm->nRanks),
  };
  const cocclPipelineSpec spec = {
      "reducescatter-oneshot", args.sendbuff, args.recvbuff,
      args.count, (size_t)comm->nRanks, args.datatype, comm,
      prepared->compressor, args.stream, stages,
      (int)(sizeof(stages) / sizeof(stages[0])),
      cocclPipelineInPlaceOutputRankChunk};
  return cocclRunPipeline(&spec);
}

ncclResult_t cocclRunReduceScatterTwoShot(
    const cocclPreparedCall* prepared) {
  const cocclInfo& args = prepared->info;
  ncclComm_t comm = args.comm;
  const int localRanks = comm->localRanks;
  const int nNodes = comm->nRanks / localRanks;
  cocclHierarchicalComms hierarchy = {};
  NCCLCHECK(cocclCommGetHierarchicalComms(comm, &hierarchy));

  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(),
      cocclPipelineAllToAll(hierarchy.intraComm),
      cocclPipelineDecompReduceComp((size_t)localRanks),
      cocclPipelineAllToAll(hierarchy.interComm),
      cocclPipelineDecompressReduce((size_t)nNodes),
  };
  const cocclPipelineSpec spec = {
      "reducescatter-twoshot", args.sendbuff, args.recvbuff,
      args.count, (size_t)comm->nRanks, args.datatype, comm,
      prepared->compressor, args.stream, stages,
      (int)(sizeof(stages) / sizeof(stages[0])),
      cocclPipelineInPlaceOutputRankChunk};
  return cocclRunPipeline(&spec);
}

cocclPreparedCall cocclDirectReduceScatterCall(
    const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    cudaStream_t stream, cocclAlgorithmKind algorithm,
    const cocclCompressorHandle& compressor) {
  cocclPreparedCall prepared;
  prepared.info.sendbuff = sendbuff;
  prepared.info.recvbuff = recvbuff;
  prepared.info.count = recvcount;
  prepared.info.datatype = datatype;
  prepared.info.op = op;
  prepared.info.func = ncclFuncReduceScatter;
  prepared.info.operation = cocclOperation::ReduceScatter;
  prepared.info.comm = comm;
  prepared.info.stream = stream;
  prepared.descriptor =
      cocclGetOperationDescriptor(cocclOperation::ReduceScatter);
  prepared.policy = algorithm == cocclAlgorithmReduceScatterTwoShot
      ? cocclHierarchicalPolicy(cocclOperation::ReduceScatter)
      : cocclDefaultPolicy(cocclOperation::ReduceScatter);
  prepared.algorithm = algorithm;
  prepared.compressor = compressor;
  return prepared;
}

}  // namespace

NCCL_API(ncclResult_t, cocclReduceScatterCompOneShot,
  const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm,
  cudaStream_t stream);
ncclResult_t cocclReduceScatterCompOneShot(const void* sendbuff,
  void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op,
  ncclComm* comm, cudaStream_t stream) {
  cocclCompressorHandle compressor;
  NCCLCHECK(cocclCommGetCompressor(
      comm, cocclDefaultPolicy(cocclOperation::ReduceScatter), &compressor));
  cocclPreparedCall prepared = cocclDirectReduceScatterCall(
      sendbuff, recvbuff, recvcount, datatype, op, comm, stream,
      cocclAlgorithmReduceScatterOneShot, compressor);
  return cocclExecutePreparedCall(&prepared);
}

NCCL_API(ncclResult_t, cocclReduceScatterCompTwoShot,
  const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm,
  cudaStream_t stream);
ncclResult_t cocclReduceScatterCompTwoShot(const void* sendbuff,
  void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op,
  ncclComm* comm, cudaStream_t stream) {
  cocclCompressorHandle compressor;
  NCCLCHECK(cocclCommGetCompressor(
      comm, cocclHierarchicalPolicy(cocclOperation::ReduceScatter),
      &compressor));
  cocclPreparedCall prepared = cocclDirectReduceScatterCall(
      sendbuff, recvbuff, recvcount, datatype, op, comm, stream,
      cocclAlgorithmReduceScatterTwoShot, compressor);
  return cocclExecutePreparedCall(&prepared);
}

ncclResult_t cocclExecuteReduceScatter(const cocclPreparedCall* prepared) {
  const ncclComm_t comm = prepared->info.comm;
  if (prepared->algorithm == cocclAlgorithmReduceScatterTwoShot &&
      (comm->localRanks <= 0 || comm->nRanks % comm->localRanks != 0)) {
    return ncclInvalidArgument;
  }
  switch (prepared->algorithm) {
    case cocclAlgorithmReduceScatterOneShot:
      return cocclRunReduceScatterOneShot(prepared);
    case cocclAlgorithmReduceScatterTwoShot:
      return cocclRunReduceScatterTwoShot(prepared);
    default:
      return ncclInternalError;
  }
}
