#include "compress.h"

#include "coccl_buffer_management.h"
#include "coccl_comm.h"
#include "collectives.h"
#include "info.h"
#include "reduce_extend.h"

namespace {

// compress.cc is deliberately execution-only: coccl_comm owns plugin loading,
// chain selection, and config lifetime. This file only invokes callbacks.
static ncclResult_t checkCompressor(ncclCompressor_t* compressor) {
  if (compressor == nullptr) {
    WARN("COCCL compressor chain contains a null compressor");
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

struct CompressContext {
  const void* orgbuff;
  void** compbuff;
  size_t orgChunkCount;
  ncclDataType_t orgDatatype;
  size_t* compChunkCount;
  ncclDataType_t* compDatatype;
  size_t numChunks;
  int rank;
  cudaStream_t stream;
};

static ncclResult_t compressVisitor(ncclCompressor_t* compressor, void* config, void* context) {
  NCCLCHECK(checkCompressor(compressor));
  CompressContext* args = static_cast<CompressContext*>(context);
  CUDACHECK(compressor->compress(args->orgbuff, args->compbuff, args->orgChunkCount,
                                 args->orgDatatype, args->compChunkCount, args->compDatatype,
                                 args->numChunks, args->rank, config, NULL, args->stream));
  return ncclSuccess;
}

static ncclResult_t runCompressChain(ncclComm_t comm, ncclCommOp_t commOp, const void* orgbuff,
                                     void** compbuff, const size_t orgChunkCount,
                                     ncclDataType_t orgDatatype, size_t* compChunkCount,
                                     ncclDataType_t* compDatatype, const size_t numChunks,
                                     const int rank, cudaStream_t stream) {
  CompressContext context = {
      orgbuff, compbuff, orgChunkCount, orgDatatype, compChunkCount, compDatatype,
      numChunks, rank, stream};
  // Compression follows env order so multi-stage compressors compose naturally.
  return cocclVisitCompressorChain(comm, commOp, false, compressVisitor, &context);
}

struct DecompressContext {
  void* decompbuff;
  const void* compbuff;
  size_t decompChunkCount;
  ncclDataType_t decompDatatype;
  size_t compChunkCount;
  ncclDataType_t compDatatype;
  size_t numChunks;
  cudaStream_t stream;
};

static ncclResult_t decompressVisitor(ncclCompressor_t* compressor, void* config, void* context) {
  NCCLCHECK(checkCompressor(compressor));
  DecompressContext* args = static_cast<DecompressContext*>(context);
  CUDACHECK(compressor->decompress(args->decompbuff, args->compbuff, args->decompChunkCount,
                                   args->decompDatatype, args->compChunkCount, args->compDatatype,
                                   args->numChunks, config, args->stream));
  return ncclSuccess;
}

static ncclResult_t runDecompressChain(ncclComm_t comm, ncclCommOp_t commOp, void* decompbuff,
                                       const void* compbuff, const size_t decompChunkCount,
                                       ncclDataType_t decompDatatype, const size_t compChunkCount,
                                       ncclDataType_t compDatatype, const size_t numChunks,
                                       cudaStream_t stream) {
  DecompressContext context = {
      decompbuff, compbuff, decompChunkCount, decompDatatype, compChunkCount, compDatatype,
      numChunks, stream};
  // Decompression reverses the chain to undo staged compression in LIFO order.
  return cocclVisitCompressorChain(comm, commOp, true, decompressVisitor, &context);
}

// Fallback path when a compressor plugin does not provide fused decomp+reduce.
// We decompress into TLS scratch, then reuse COCCL's reduce kernel.
static ncclResult_t cocclDecompReduce(void* reducebuff, const void* compbuff,
                                      const size_t compChunkCount, ncclDataType_t compDatatype,
                                      const size_t reduceChunkCount, ncclDataType_t reduceDataType,
                                      const size_t numChunks, ncclCompressor_t* compressor,
                                      void* config, ncclComm_t comm, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  cocclBufferHandle reduceBuffer = {};
  void* reduceWorkspace = nullptr;
  NCCLCHECKGOTO(cocclGetBuffer(comm, reduceChunkCount * numChunks * ncclTypeSize(reduceDataType),
                               &reduceBuffer),
                ret, exit);
  reduceWorkspace = reduceBuffer.ptr;

  CUDACHECKGOTO(compressor->decompress(reduceWorkspace, compbuff, reduceChunkCount, reduceDataType,
                                       compChunkCount, compDatatype, numChunks, config, stream),
                ret, exit);

  NCCLCHECKGOTO(ncclReduceChunk(reduceWorkspace, reduceChunkCount, reducebuff, reduceDataType, numChunks, stream),
                ret, exit);

exit:
  if (reduceBuffer.ptr != nullptr) {
    ncclResult_t releaseRet = cocclReleaseBuffer(&reduceBuffer, stream);
    if (ret == ncclSuccess) ret = releaseRet;
  }
  return ret;
}

struct DecompressReduceContext {
  void* reducebuff;
  const void* compbuff;
  size_t compChunkCount;
  ncclDataType_t compDatatype;
  size_t reduceChunkCount;
  ncclDataType_t reduceDataType;
  size_t numChunks;
  ncclComm_t comm;
  cudaStream_t stream;
};

static ncclResult_t decompressReduceVisitor(ncclCompressor_t* compressor, void* config, void* context) {
  NCCLCHECK(checkCompressor(compressor));
  DecompressReduceContext* args = static_cast<DecompressReduceContext*>(context);
  // Prefer plugin fused callback when available; fall back to explicit
  // decompress + reduce to preserve behavior for simpler compressors.
  if (compressor->decompReduce == nullptr) {
    NCCLCHECK(cocclDecompReduce(args->reducebuff, args->compbuff, args->compChunkCount,
                                args->compDatatype, args->reduceChunkCount,
                                args->reduceDataType, args->numChunks, compressor,
                                config, args->comm, args->stream));
  } else {
    CUDACHECK(compressor->decompReduce(args->reducebuff, args->compbuff, args->compChunkCount,
                                       args->compDatatype, args->reduceChunkCount,
                                       args->reduceDataType, args->numChunks, config,
                                       args->stream));
  }
  return ncclSuccess;
}

static ncclResult_t runDecompressReduceChain(ncclComm_t comm, ncclCommOp_t commOp, void* reducebuff,
                                             const void* compbuff, const size_t compChunkCount,
                                             ncclDataType_t compDatatype, const size_t reduceChunkCount,
                                             ncclDataType_t reduceDataType, const size_t numChunks,
                                             cudaStream_t stream) {
  DecompressReduceContext context = {
      reducebuff, compbuff, compChunkCount, compDatatype, reduceChunkCount, reduceDataType,
      numChunks, comm, stream};
  return cocclVisitCompressorChain(comm, commOp, true, decompressReduceVisitor, &context);
}

static ncclResult_t cocclDecompReduceComp(const void* compbuff, void** recompbuff,
                                          const size_t orgChunkCount, ncclDataType_t orgDatatype,
                                          const size_t compChunkCount, ncclDataType_t compDatatype,
                                          size_t* reCompChunkCount, ncclDataType_t* reCompDatatype,
                                          const size_t numChunks, ncclCompressor_t* compressor,
                                          void* config, ncclComm_t comm, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  cocclBufferHandle reduceBuffer = {};
  void* reduceWorkspace = nullptr;
  // Fallback for reduce-scatter style paths where the next stage still expects
  // compressed data: decompress, reduce, then compress the reduced result.
  NCCLCHECKGOTO(cocclGetBuffer(comm, orgChunkCount * numChunks * ncclTypeSize(orgDatatype),
                               &reduceBuffer),
                ret, exit);
  reduceWorkspace = reduceBuffer.ptr;

  CUDACHECKGOTO(compressor->decompress(reduceWorkspace, compbuff, orgChunkCount, orgDatatype,
                                       compChunkCount, compDatatype, numChunks, config, stream),
                ret, exit);

  NCCLCHECKGOTO(ncclReduceChunk(reduceWorkspace, orgChunkCount, reduceWorkspace, orgDatatype, numChunks, stream),
                ret, exit);

  CUDACHECKGOTO(compressor->compress(reduceWorkspace, recompbuff, orgChunkCount, orgDatatype,
                                     reCompChunkCount, reCompDatatype, 1, 0, config,
                                     NULL, stream),
                ret, exit);

exit:
  if (reduceBuffer.ptr != nullptr) {
    ncclResult_t releaseRet = cocclReleaseBuffer(&reduceBuffer, stream);
    if (ret == ncclSuccess) ret = releaseRet;
  }
  return ret;
}

struct DecompressReduceCompressContext {
  const void* compbuff;
  void** recompbuff;
  size_t orgChunkCount;
  ncclDataType_t orgDatatype;
  size_t compChunkCount;
  ncclDataType_t compDatatype;
  size_t* reCompChunkCount;
  ncclDataType_t* reCompDatatype;
  size_t numChunks;
  ncclComm_t comm;
  cudaStream_t stream;
};

static ncclResult_t decompressReduceCompressVisitor(ncclCompressor_t* compressor, void* config, void* context) {
  NCCLCHECK(checkCompressor(compressor));
  DecompressReduceCompressContext* args = static_cast<DecompressReduceCompressContext*>(context);
  // The fused plugin path avoids materializing the reduced uncompressed
  // buffer; the fallback keeps the same semantics for plugins without it.
  if (compressor->decompReduceComp == nullptr) {
    NCCLCHECK(cocclDecompReduceComp(args->compbuff, args->recompbuff, args->orgChunkCount,
                                    args->orgDatatype, args->compChunkCount, args->compDatatype,
                                    args->reCompChunkCount, args->reCompDatatype,
                                    args->numChunks, compressor, config, args->comm, args->stream));
  } else {
    CUDACHECK(compressor->decompReduceComp(args->compbuff, args->recompbuff, args->compChunkCount,
                                           args->compDatatype, args->reCompChunkCount,
                                           args->reCompDatatype, args->numChunks, config,
                                           NULL, args->stream));
  }
  return ncclSuccess;
}

static ncclResult_t runDecompressReduceCompressChain(ncclComm_t comm, ncclCommOp_t commOp,
                                                     const void* compbuff, void** recompbuff,
                                                     const size_t orgChunkCount,
                                                     ncclDataType_t orgDatatype,
                                                     const size_t compChunkCount,
                                                     ncclDataType_t compDatatype,
                                                     size_t* reCompChunkCount,
                                                     ncclDataType_t* reCompDatatype,
                                                     const size_t numChunks,
                                                     cudaStream_t stream) {
  DecompressReduceCompressContext context = {
      compbuff, recompbuff, orgChunkCount, orgDatatype, compChunkCount, compDatatype,
      reCompChunkCount, reCompDatatype, numChunks, comm, stream};
  return cocclVisitCompressorChain(comm, commOp, true, decompressReduceCompressVisitor, &context);
}

}

// These symbols are still COCCL-private. They lookup the chain by communicator
// and op tag, then delegate actual work to the helpers above.
ncclResult_t ncclCompress(const void* orgbuff, void** compbuff, const size_t orgChunkCount,
                          ncclDataType_t orgDatatype, size_t* compChunkCount,
                          ncclDataType_t* compDatatype, const size_t numChunks, const int rank,
                          ncclComm_t comm, ncclCommOp_t commOp, cudaStream_t stream) {
  return runCompressChain(comm, commOp, orgbuff, compbuff, orgChunkCount, orgDatatype,
                          compChunkCount, compDatatype, numChunks, rank, stream);
}

ncclResult_t ncclDecompress(void* decompbuff, const void* compbuff, const size_t decompChunkCount,
                            ncclDataType_t decompDatatype, const size_t compChunkCount,
                            ncclDataType_t compDatatype, const size_t numChunks,
                            ncclComm_t comm, ncclCommOp_t commOp, cudaStream_t stream) {
  return runDecompressChain(comm, commOp, decompbuff, compbuff, decompChunkCount,
                            decompDatatype, compChunkCount, compDatatype, numChunks, stream);
}

ncclResult_t ncclDecompressReduce(void* reducebuff, const void* compbuff,
                                  const size_t compChunkCount, ncclDataType_t compDatatype,
                                  const size_t reduceChunkCount, ncclDataType_t reduceDataType,
                                  const size_t numChunks, ncclComm_t comm, ncclCommOp_t commOp,
                                  cudaStream_t stream) {
  return runDecompressReduceChain(comm, commOp, reducebuff, compbuff, compChunkCount,
                                  compDatatype, reduceChunkCount, reduceDataType,
                                  numChunks, stream);
}

ncclResult_t ncclDecompReduceComp(const void* compbuff, void** recompbuff,
                                  const size_t orgChunkCount, ncclDataType_t orgDatatype,
                                  const size_t compChunkCount, ncclDataType_t compDatatype,
                                  size_t* reCompChunkCount, ncclDataType_t* reCompDatatype,
                                  const size_t numChunks, ncclComm_t comm, ncclCommOp_t commOp,
                                  cudaStream_t stream) {
  return runDecompressReduceCompressChain(comm, commOp, compbuff, recompbuff,
                                          orgChunkCount, orgDatatype, compChunkCount,
                                          compDatatype, reCompChunkCount, reCompDatatype,
                                          numChunks, stream);
}
