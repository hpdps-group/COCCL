#include "primitives/coccl_primitives_internal.h"

namespace {

ncclResult_t cocclValidateAllReducePartition(const cocclInfo& args) {
  return args.comm == nullptr || args.comm->nRanks <= 0 ||
                 args.count % (size_t)args.comm->nRanks != 0
      ? ncclInvalidArgument
      : ncclSuccess;
}

ncclResult_t cocclRunAllReduceOneShot(const cocclPreparedCall* prepared) {
  const cocclInfo& args = prepared->info;
  NCCLCHECK(cocclValidateAllReducePartition(args));
  const size_t chunkCount = args.count / (size_t)args.comm->nRanks;
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(),
      cocclPipelineAllGather(args.comm),
      cocclPipelineDecompressReduce((size_t)args.comm->nRanks),
  };
  const cocclPipelineSpec spec = {
      "allreduce-oneshot", args.sendbuff, args.recvbuff, chunkCount,
      (size_t)args.comm->nRanks, args.datatype, args.comm,
      prepared->compressor, args.stream, stages,
      (int)(sizeof(stages) / sizeof(stages[0]))};
  return cocclRunPipelineSerial(&spec);
}

ncclResult_t cocclRunAllReduceTwoShot(const cocclPreparedCall* prepared) {
  const cocclInfo& args = prepared->info;
  NCCLCHECK(cocclValidateAllReducePartition(args));
  const size_t recvcount = args.count / (size_t)args.comm->nRanks;
  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(),
      cocclPipelineAllToAll(args.comm),
      cocclPipelineDecompReduceComp((size_t)args.comm->nRanks),
      cocclPipelineAllGather(args.comm),
      cocclPipelineDecompress(),
  };
  const cocclPipelineSpec spec = {
      "allreduce-twoshot-overlap", args.sendbuff, args.recvbuff, recvcount,
      (size_t)args.comm->nRanks, args.datatype, args.comm,
      prepared->compressor, args.stream, stages,
      (int)(sizeof(stages) / sizeof(stages[0]))};
  return cocclRunPipeline(&spec);
}

ncclResult_t cocclRunAllReduceTripleShot(const cocclPreparedCall* prepared) {
  const cocclInfo& args = prepared->info;
  NCCLCHECK(cocclValidateAllReducePartition(args));
  const int localRanks = args.comm->localRanks;
  if (localRanks <= 0 || args.comm->nRanks % localRanks != 0) {
    return ncclInvalidArgument;
  }
  const size_t recvcount = args.count / (size_t)args.comm->nRanks;
  const int nNodes = args.comm->nRanks / localRanks;
  cocclHierarchicalComms hierarchy = {};
  NCCLCHECK(cocclCommGetHierarchicalComms(args.comm, &hierarchy));

  const cocclPipelineStage stages[] = {
      cocclPipelineCompress(),
      cocclPipelineAllToAll(hierarchy.intraComm),
      cocclPipelineDecompReduceComp((size_t)localRanks),
      cocclPipelineAllToAll(hierarchy.interComm),
      cocclPipelineDecompReduceComp((size_t)nNodes),
      cocclPipelineAllGather(args.comm),
      cocclPipelineDecompress(),
  };
  const cocclPipelineSpec spec = {
      "allreduce-tripleshot-tl-overlap", args.sendbuff, args.recvbuff,
      recvcount, (size_t)args.comm->nRanks, args.datatype, args.comm,
      prepared->compressor, args.stream, stages,
      (int)(sizeof(stages) / sizeof(stages[0]))};
  return cocclRunPipeline(&spec);
}

cocclPreparedCall cocclDirectAllReduceCall(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    cudaStream_t stream, cocclAlgorithmKind algorithm,
    const cocclCompressorHandle& compressor) {
  cocclPreparedCall prepared;
  prepared.info.sendbuff = sendbuff;
  prepared.info.recvbuff = recvbuff;
  prepared.info.count = count;
  prepared.info.datatype = datatype;
  prepared.info.op = op;
  prepared.info.func = ncclFuncAllReduce;
  prepared.info.operation = cocclOperation::AllReduce;
  prepared.info.comm = comm;
  prepared.info.stream = stream;
  prepared.descriptor =
      cocclGetOperationDescriptor(cocclOperation::AllReduce);
  prepared.policy = algorithm == cocclAlgorithmAllReduceTripleShot
      ? cocclHierarchicalPolicy(cocclOperation::AllReduce)
      : cocclDefaultPolicy(cocclOperation::AllReduce);
  prepared.algorithm = algorithm;
  prepared.compressor = compressor;
  return prepared;
}

}  // namespace

NCCL_API(ncclResult_t, ncclAllReduceCompOneShot, const void* sendbuff,
  void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
  ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceCompOneShot(const void* sendbuff, void* recvbuff,
  size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm,
  cudaStream_t stream) {
  cocclCompressorHandle compressor;
  NCCLCHECK(cocclCommGetCompressor(
      comm, cocclDefaultPolicy(cocclOperation::AllReduce), &compressor));
  cocclPreparedCall prepared = cocclDirectAllReduceCall(
      sendbuff, recvbuff, count, datatype, op, comm, stream,
      cocclAlgorithmAllReduceOneShot, compressor);
  return cocclExecutePreparedCall(&prepared);
}

NCCL_API(ncclResult_t, ncclAllReduceCompTwoShotOverlap,
  const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm,
  cudaStream_t stream);
ncclResult_t ncclAllReduceCompTwoShotOverlap(const void* sendbuff,
  void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
  ncclComm* comm, cudaStream_t stream) {
  cocclCompressorHandle compressor;
  NCCLCHECK(cocclCommGetCompressor(
      comm, cocclDefaultPolicy(cocclOperation::AllReduce), &compressor));
  cocclPreparedCall prepared = cocclDirectAllReduceCall(
      sendbuff, recvbuff, count, datatype, op, comm, stream,
      cocclAlgorithmAllReduceTwoShot, compressor);
  return cocclExecutePreparedCall(&prepared);
}

NCCL_API(ncclResult_t, ncclAllReduceCompTripleShotTLOverlap,
  const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm,
  cudaStream_t stream);
ncclResult_t ncclAllReduceCompTripleShotTLOverlap(const void* sendbuff,
  void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
  ncclComm* comm, cudaStream_t stream) {
  cocclCompressorHandle compressor;
  NCCLCHECK(cocclCommGetCompressor(
      comm, cocclHierarchicalPolicy(cocclOperation::AllReduce), &compressor));
  cocclPreparedCall prepared = cocclDirectAllReduceCall(
      sendbuff, recvbuff, count, datatype, op, comm, stream,
      cocclAlgorithmAllReduceTripleShot, compressor);
  return cocclExecutePreparedCall(&prepared);
}

ncclResult_t cocclExecuteAllReduce(const cocclPreparedCall* prepared) {
  if (prepared == nullptr || prepared->info.comm == nullptr ||
      !prepared->compressor) {
    return ncclInvalidArgument;
  }
  switch (prepared->algorithm) {
    case cocclAlgorithmAllReduceOneShot:
      return cocclRunAllReduceOneShot(prepared);
    case cocclAlgorithmAllReduceTwoShot:
      return cocclRunAllReduceTwoShot(prepared);
    case cocclAlgorithmAllReduceTripleShot:
      return cocclRunAllReduceTripleShot(prepared);
    default:
      return ncclInternalError;
  }
}
