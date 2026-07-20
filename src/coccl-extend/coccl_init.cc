#include "coccl_init.h"

#include "coccl_autotune.h"
#include "coccl_buffer_management.h"
#include "coccl_comm.h"
#include "coccl_training_assist.h"
#include "comm.h"
#include "compress_utils.h"
#include "nccl_common.h"
#include "param.h"

#include <dlfcn.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <unordered_map>
#include <vector>

constexpr int64_t kDefaultCompressionThresholdBytes = 8LL * 1024 * 1024;

NCCL_PARAM(AllToAllCompEnableThreshold,
           "ALLTOALL_COMPRESS_ENABLE_THRESHOLD",
           kDefaultCompressionThresholdBytes);
NCCL_PARAM(AllReduceCompEnableThreshold,
           "ALLREDUCE_COMPRESS_ENABLE_THRESHOLD",
           kDefaultCompressionThresholdBytes);
NCCL_PARAM(AllGatherCompEnableThreshold,
           "ALLGATHER_COMPRESS_ENABLE_THRESHOLD",
           kDefaultCompressionThresholdBytes);
NCCL_PARAM(ReduceScatterCompEnableThreshold,
           "REDUCESCATTER_COMPRESS_ENABLE_THRESHOLD",
           kDefaultCompressionThresholdBytes);
NCCL_PARAM(SendRecvCompEnableThreshold,
           "SENDRECV_COMPRESS_ENABLE_THRESHOLD",
           kDefaultCompressionThresholdBytes);

NCCL_PARAM(DpAllReduceCompEnableThreshold,
           "DP_ALLREDUCE_COMPRESS_ENABLE_THRESHOLD",
           kDefaultCompressionThresholdBytes);
NCCL_PARAM(DpAllGatherCompEnableThreshold,
           "DP_ALLGATHER_COMPRESS_ENABLE_THRESHOLD",
           kDefaultCompressionThresholdBytes);
NCCL_PARAM(DpReduceScatterCompEnableThreshold,
           "DP_REDUCESCATTER_COMPRESS_ENABLE_THRESHOLD",
           kDefaultCompressionThresholdBytes);
NCCL_PARAM(TpAllReduceCompEnableThreshold,
           "TP_ALLREDUCE_COMPRESS_ENABLE_THRESHOLD",
           kDefaultCompressionThresholdBytes);
NCCL_PARAM(TpAllGatherCompEnableThreshold,
           "TP_ALLGATHER_COMPRESS_ENABLE_THRESHOLD",
           kDefaultCompressionThresholdBytes);
NCCL_PARAM(TpReduceScatterCompEnableThreshold,
           "TP_REDUCESCATTER_COMPRESS_ENABLE_THRESHOLD",
           kDefaultCompressionThresholdBytes);
NCCL_PARAM(PpSendRecvCompEnableThreshold,
           "PP_SENDRECV_COMPRESS_ENABLE_THRESHOLD",
           kDefaultCompressionThresholdBytes);

namespace {

using cocclThresholdParam = int64_t (*)();

// One row is the complete definition of a compressor operation. The table is
// deliberately explicit because env names and config suffixes use historical
// spellings that cannot be derived safely from the enum.
struct cocclCompressorOpMetadata {
  ncclCommOp_t op;
  ncclCommOp_t baseOp;
  const char* enableEnvName;
  const char* compressorsEnvName;
  const char* configSuffix;
  cocclThresholdParam thresholdParam;
};

constexpr cocclCompressorOpMetadata kCompressorOps[] = {
    {AlltoAll, AlltoAll, "NCCL_ENABLE_ALLTOALL_COMPRESS",
     "NCCL_ALLTOALL_COMPRESSORS", "A2A",
     ncclParamAllToAllCompEnableThreshold},
    {AlltoAll_Inter, AlltoAll, "NCCL_ENABLE_ALLTOALL_COMPRESS",
     "NCCL_ALLTOALL_INTER_COMPRESSORS", "A2A_Inter",
     ncclParamAllToAllCompEnableThreshold},
    {AllReduce, AllReduce, "NCCL_ENABLE_ALLREDUCE_COMPRESS",
     "NCCL_ALLREDUCE_COMPRESSORS", "AR",
     ncclParamAllReduceCompEnableThreshold},
    {AllReduce_Inter, AllReduce, "NCCL_ENABLE_ALLREDUCE_COMPRESS",
     "NCCL_ALLREDUCE_INTER_COMPRESSORS", "AR_Inter",
     ncclParamAllReduceCompEnableThreshold},
    {AllGather, AllGather, "NCCL_ENABLE_ALLGATHER_COMPRESS",
     "NCCL_ALLGATHER_COMPRESSORS", "AG",
     ncclParamAllGatherCompEnableThreshold},
    {AllGather_Inter, AllGather, "NCCL_ENABLE_ALLGATHER_COMPRESS",
     "NCCL_ALLGATHER_INTER_COMPRESSORS", "AG_Inter",
     ncclParamAllGatherCompEnableThreshold},
    {ReduceScatter, ReduceScatter, "NCCL_ENABLE_REDUCESCATTER_COMPRESS",
     "NCCL_REDUCESCATTER_COMPRESSORS", "RS",
     ncclParamReduceScatterCompEnableThreshold},
    {ReduceScatter_Inter, ReduceScatter,
     "NCCL_ENABLE_REDUCESCATTER_COMPRESS",
     "NCCL_REDUCESCATTER_INTER_COMPRESSORS", "RS_Inter",
     ncclParamReduceScatterCompEnableThreshold},
    {SendRecv, SendRecv, "NCCL_ENABLE_SENDRECV_COMPRESS",
     "NCCL_SENDRECV_COMPRESSORS", "SR",
     ncclParamSendRecvCompEnableThreshold},
    {SendRecv_BWD, SendRecv, "NCCL_ENABLE_SENDRECV_COMPRESS",
     "NCCL_SENDRECV_BWD_COMPRESSORS", "SR_BWD",
     ncclParamSendRecvCompEnableThreshold},
};

struct cocclTrainingRoleCompressorMetadata {
  cocclTrainingRole role;
  const char* enableEnvName;
};

struct cocclTrainingRoleOpCompressorMetadata {
  cocclTrainingRole role;
  ncclCommOp_t op;
  const char* compressorsEnvName;
  cocclThresholdParam thresholdParam;
};

constexpr cocclTrainingRoleCompressorMetadata kTrainingRoles[] = {
    {cocclTrainingRoleDataParallel, "NCCL_ENABLE_DP_COMPRESS"},
    {cocclTrainingRolePipelineParallel, "NCCL_ENABLE_PP_COMPRESS"},
    {cocclTrainingRoleTensorParallel, "NCCL_ENABLE_TP_COMPRESS"},
};

// Training mode selects compressors by both the detected parallelism role and
// operation. Inter stages use their parent primitive's list but retain the
// operation metadata's distinct config suffix.
constexpr cocclTrainingRoleOpCompressorMetadata
    kTrainingRoleOpCompressors[] = {
        {cocclTrainingRoleDataParallel, AllReduce,
         "NCCL_DP_ALLREDUCE_COMPRESSORS",
         ncclParamDpAllReduceCompEnableThreshold},
        {cocclTrainingRoleDataParallel, AllReduce_Inter,
         "NCCL_DP_ALLREDUCE_COMPRESSORS",
         ncclParamDpAllReduceCompEnableThreshold},
        {cocclTrainingRoleDataParallel, AllGather,
         "NCCL_DP_ALLGATHER_COMPRESSORS",
         ncclParamDpAllGatherCompEnableThreshold},
        {cocclTrainingRoleDataParallel, AllGather_Inter,
         "NCCL_DP_ALLGATHER_COMPRESSORS",
         ncclParamDpAllGatherCompEnableThreshold},
        {cocclTrainingRoleDataParallel, ReduceScatter,
         "NCCL_DP_REDUCESCATTER_COMPRESSORS",
         ncclParamDpReduceScatterCompEnableThreshold},
        {cocclTrainingRoleDataParallel, ReduceScatter_Inter,
         "NCCL_DP_REDUCESCATTER_COMPRESSORS",
         ncclParamDpReduceScatterCompEnableThreshold},
        {cocclTrainingRoleTensorParallel, AllReduce,
         "NCCL_TP_ALLREDUCE_COMPRESSORS",
         ncclParamTpAllReduceCompEnableThreshold},
        {cocclTrainingRoleTensorParallel, AllReduce_Inter,
         "NCCL_TP_ALLREDUCE_COMPRESSORS",
         ncclParamTpAllReduceCompEnableThreshold},
        {cocclTrainingRoleTensorParallel, AllGather,
         "NCCL_TP_ALLGATHER_COMPRESSORS",
         ncclParamTpAllGatherCompEnableThreshold},
        {cocclTrainingRoleTensorParallel, AllGather_Inter,
         "NCCL_TP_ALLGATHER_COMPRESSORS",
         ncclParamTpAllGatherCompEnableThreshold},
        {cocclTrainingRoleTensorParallel, ReduceScatter,
         "NCCL_TP_REDUCESCATTER_COMPRESSORS",
         ncclParamTpReduceScatterCompEnableThreshold},
        {cocclTrainingRoleTensorParallel, ReduceScatter_Inter,
         "NCCL_TP_REDUCESCATTER_COMPRESSORS",
         ncclParamTpReduceScatterCompEnableThreshold},
        {cocclTrainingRolePipelineParallel, SendRecv,
         "NCCL_PP_SENDRECV_FWD_COMPRESSORS",
         ncclParamPpSendRecvCompEnableThreshold},
        {cocclTrainingRolePipelineParallel, SendRecv_BWD,
         "NCCL_PP_SENDRECV_BWD_COMPRESSORS",
         ncclParamPpSendRecvCompEnableThreshold},
};

pthread_mutex_t cocclInitLock = PTHREAD_MUTEX_INITIALIZER;

// Compressor libraries are process-level objects. Keep dlopen handles alive
// for the process lifetime to avoid plugin state teardown races.
bool compressorHandlesLoaded = false;
std::unordered_map<std::string, ncclCompressor_t*> compressorHandles;

bool envEnabled(const char* name) {
  const char* value = getenv(name);
  return value != nullptr && strcmp(value, "1") == 0;
}

size_t compressionThreshold(cocclThresholdParam thresholdParam) {
  const int64_t value = thresholdParam == nullptr ? 0 : thresholdParam();
  return value > 0 ? (size_t)value : 0;
}

const cocclCompressorOpMetadata* compressorMetadataForOp(ncclCommOp_t op) {
  for (const cocclCompressorOpMetadata& metadata : kCompressorOps) {
    if (metadata.op == op) return &metadata;
  }
  return nullptr;
}

const cocclTrainingRoleCompressorMetadata* trainingMetadataForRole(
    cocclTrainingRole role) {
  for (const cocclTrainingRoleCompressorMetadata& metadata : kTrainingRoles) {
    if (metadata.role == role) return &metadata;
  }
  return nullptr;
}

void markAutotuneProfileOperation(
    ncclCommOp_t op, cocclAutotuneProfileOptions* profileOptions) {
  if (profileOptions == nullptr) return;
  const cocclCompressorOpMetadata* metadata = compressorMetadataForOp(op);
  if (metadata == nullptr) return;

  if (metadata->baseOp == ReduceScatter) {
    profileOptions->profileReduceScatter = true;
  } else if (metadata->baseOp == AllReduce) {
    profileOptions->profileAllReduce = true;
  }
}

ncclCompressor_t* findCompressorHandle(const std::string& compName) {
  auto handle = compressorHandles.find(compName);
  return handle == compressorHandles.end() ? nullptr : handle->second;
}

ncclResult_t loadCompressorHandlesLocked() {
  if (compressorHandlesLoaded) return ncclSuccess;

  // NCCL_COMPRESSORS is the global plugin allow-list and load order. Every
  // operation-specific chain may only reference names loaded from this list.
  const char* usedComp = getenv("NCCL_COMPRESSORS");
  const char* compLibPath = getenv("NCCL_COMPRESSORS_LIB_PATH");
  for (const std::string& compName : parseCompressorNames(usedComp)) {
    if (findCompressorHandle(compName) != nullptr) continue;

    std::string compLibName = buildCompressorLibPath(compLibPath, compName);
    void* compLibHandle = tryOpenCompressorLib(compLibName.c_str());
    if (compLibHandle == nullptr) {
      WARN("COCCL failed to open compressor library %s", compLibName.c_str());
      return ncclSystemError;
    }

    ncclCompressor_t* compressor =
        (ncclCompressor_t*)dlsym(compLibHandle, compName.c_str());
    if (compressor == nullptr) {
      WARN("COCCL failed to find compressor symbol %s in %s",
           compName.c_str(), compLibName.c_str());
      return ncclSystemError;
    }
    compressorHandles[compName] = compressor;
  }

  compressorHandlesLoaded = true;
  return ncclSuccess;
}

ncclResult_t loadProfilingCompressorChain(ncclComm_t comm) {
  // Before cocclComm is committed, autotune has no classified training role.
  // Keep an unconfigured copy of the global plugin list solely for profiling;
  // committed communication never falls back to this chain.
  for (const std::string& compName :
       parseCompressorNames(getenv("NCCL_COMPRESSORS"))) {
    ncclCompressor_t* compressor = findCompressorHandle(compName);
    if (compressor == nullptr) {
      WARN("COCCL compressor %s was requested by NCCL_COMPRESSORS but was not loaded",
           compName.c_str());
      return ncclSystemError;
    }

    void* unusedConfig = nullptr;
    compressor->parseConfig(nullptr, &unusedConfig, comm->nNodes,
                            comm->localRanks);
    free(unusedConfig);
    NCCLCHECK(cocclCommAppendDefaultCompressor(
        comm, compressor, nullptr));
  }
  return ncclSuccess;
}

ncclResult_t loadNormalCompressorChain(
    ncclComm_t comm, const cocclCompressorOpMetadata& metadata,
    const std::vector<std::string>& compressorNames,
    const char* configPathBase) {
  NCCLCHECK(cocclCommResetOpChain(
      comm, metadata.op, compressionThreshold(metadata.thresholdParam)));
  for (const std::string& compName : compressorNames) {
    ncclCompressor_t* compressor = findCompressorHandle(compName);
    if (compressor == nullptr) {
      WARN("COCCL compressor %s was requested by %s but was not loaded by NCCL_COMPRESSORS",
           compName.c_str(), metadata.compressorsEnvName);
      return ncclInvalidArgument;
    }

    void* config = nullptr;
    std::string configPath = buildCompressorConfigPath(
        configPathBase, compName, metadata.configSuffix);
    // coccl_comm.cc takes ownership of every non-null config pointer.
    compressor->parseConfig(configPath.c_str(), &config, comm->nNodes,
                            comm->localRanks);
    NCCLCHECK(cocclCommAppendOpCompressor(
        comm, metadata.op, compressor, config));
  }
  return ncclSuccess;
}

ncclResult_t loadNormalCompressorChains(
    ncclComm_t comm, cocclAutotuneProfileOptions* profileOptions,
    const char* configPathBase) {
  // A primitive is enabled only when both its enable switch and its base-op
  // compressor list are explicit. NCCL_COMPRESSORS remains the plugin
  // allow-list; it is no longer an implicit execution chain.
  for (const cocclCompressorOpMetadata& metadata : kCompressorOps) {
    if (metadata.op != metadata.baseOp ||
        !envEnabled(metadata.enableEnvName)) {
      continue;
    }

    std::vector<std::string> compressorNames =
        parseCompressorNames(getenv(metadata.compressorsEnvName));
    if (compressorNames.empty()) {
      INFO(NCCL_INIT,
           "COCCL op %d compression disabled because %s is unset or empty",
           (int)metadata.op, metadata.compressorsEnvName);
      continue;
    }

    NCCLCHECK(loadNormalCompressorChain(
        comm, metadata, compressorNames, configPathBase));
    markAutotuneProfileOperation(metadata.op, profileOptions);

    // Inter/BWD tags are internal stages. A separately configured chain wins;
    // otherwise they inherit the explicitly configured base chain, including
    // its threshold. No parallel primitive-enable state is needed.
    for (const cocclCompressorOpMetadata& stageMetadata : kCompressorOps) {
      if (stageMetadata.op == stageMetadata.baseOp ||
          stageMetadata.baseOp != metadata.op) {
        continue;
      }

      std::vector<std::string> stageCompressorNames =
          parseCompressorNames(getenv(stageMetadata.compressorsEnvName));
      if (stageCompressorNames.empty()) {
        NCCLCHECK(cocclCommCopyOpChain(
            comm, stageMetadata.op, metadata.op));
      } else {
        NCCLCHECK(loadNormalCompressorChain(
            comm, stageMetadata, stageCompressorNames, configPathBase));
      }
    }
  }
  return ncclSuccess;
}

bool trainingRoleCompressionEnabled(cocclTrainingRole role) {
  const cocclTrainingRoleCompressorMetadata* metadata =
      trainingMetadataForRole(role);
  return metadata != nullptr && envEnabled(metadata->enableEnvName);
}

ncclResult_t loadTrainingRoleChainForOp(
    ncclComm_t comm, cocclTrainingRole role,
    const std::vector<std::string>& compressorNames, ncclCommOp_t op,
    size_t thresholdBytes, const char* configPathBase) {
  const cocclCompressorOpMetadata* metadata = compressorMetadataForOp(op);
  const cocclCompressorOpMetadata* baseMetadata = metadata == nullptr
      ? nullptr : compressorMetadataForOp(metadata->baseOp);
  if (metadata == nullptr || baseMetadata == nullptr) {
    WARN("COCCL compressor op %d has no metadata", (int)op);
    return ncclInvalidArgument;
  }

  NCCLCHECK(cocclCommResetTrainingRoleChain(
      comm, role, op, thresholdBytes));
  for (const std::string& compressorName : compressorNames) {
    ncclCompressor_t* compressor = findCompressorHandle(compressorName);
    if (compressor == nullptr) {
      WARN("COCCL training role %s requested compressor %s, but it is not present in NCCL_COMPRESSORS",
           cocclTrainingRoleName(role), compressorName.c_str());
      return ncclInvalidArgument;
    }

    std::string configPath = buildCompressorConfigPath(
        configPathBase, compressorName, metadata->configSuffix);
    // Inter/BWD stages use their dedicated config when present, otherwise the
    // base operation's config file supplies the same fallback as normal mode.
    if (metadata->op != metadata->baseOp &&
        !compressorConfigFileExists(configPath)) {
      configPath = buildCompressorConfigPath(
          configPathBase, compressorName, baseMetadata->configSuffix);
    }

    void* compressorConfig = nullptr;
    compressor->parseConfig(configPath.c_str(), &compressorConfig,
                            comm->nNodes, comm->localRanks);
    NCCLCHECK(cocclCommAppendTrainingRoleCompressor(
        comm, role, op, compressor, compressorConfig));
  }
  return ncclSuccess;
}

ncclResult_t loadTrainingRoleCompressorChains(
    ncclComm_t comm, cocclAutotuneProfileOptions* profileOptions,
    const char* configPathBase) {
  for (const cocclTrainingRoleCompressorMetadata& roleMetadata :
       kTrainingRoles) {
    if (!trainingRoleCompressionEnabled(roleMetadata.role)) {
      INFO(NCCL_INIT,
           "COCCL training role %s compression disabled by %s; role classification remains active",
           cocclTrainingRoleName(roleMetadata.role),
           roleMetadata.enableEnvName);
    }
  }

  for (const cocclTrainingRoleOpCompressorMetadata& roleOp :
       kTrainingRoleOpCompressors) {
    if (!trainingRoleCompressionEnabled(roleOp.role)) continue;

    std::vector<std::string> compressorNames =
        parseCompressorNames(getenv(roleOp.compressorsEnvName));
    if (compressorNames.empty()) {
      INFO(NCCL_INIT,
           "COCCL training role %s op %d compression disabled because %s is unset or empty",
           cocclTrainingRoleName(roleOp.role), (int)roleOp.op,
           roleOp.compressorsEnvName);
      continue;
    }

    INFO(NCCL_INIT,
         "COCCL training role %s op %d loads compressor chain from %s",
         cocclTrainingRoleName(roleOp.role), (int)roleOp.op,
         roleOp.compressorsEnvName);
    NCCLCHECK(loadTrainingRoleChainForOp(
        comm, roleOp.role, compressorNames, roleOp.op,
        compressionThreshold(roleOp.thresholdParam), configPathBase));
    markAutotuneProfileOperation(roleOp.op, profileOptions);
  }
  return ncclSuccess;
}

ncclResult_t loadCompressorsForComm(
    ncclComm_t comm, cocclAutotuneProfileOptions* profileOptions) {
  INFO(NCCL_INIT, "Load COCCL compressors for comm %p", comm);
  NCCLCHECK(loadProfilingCompressorChain(comm));

  const char* configPathBase = getenv("NCCL_COMPRESSORS_CONFIG_PATH");
  if (cocclTrainingAssistEnabled()) {
    // Normal primitive switches and chains are ignored in training mode. An
    // enabled role must provide a non-empty role/op chain to enable that op.
    INFO(NCCL_INIT,
         "COCCL training mode ignores normal primitive NCCL_ENABLE_*_COMPRESS and NCCL_<OP>_COMPRESSORS settings");
    NCCLCHECK(loadTrainingRoleCompressorChains(
        comm, profileOptions, configPathBase));
  } else {
    NCCLCHECK(loadNormalCompressorChains(
        comm, profileOptions, configPathBase));
  }
  return ncclSuccess;
}

}  // namespace

bool cocclCompressionEnabled() {
  return envEnabled("NCCL_ENABLE_COMPRESS");
}

ncclResult_t cocclInit(ncclComm_t comm) {
  if (!cocclCompressionEnabled()) return ncclSuccess;
  if (comm == nullptr) return ncclInvalidArgument;

  ncclResult_t ret = ncclSuccess;
  cocclAutotuneProfileOptions profileOptions = {};
  bool registryEmpty = false;
  cocclCommDetachedResources* detachedResources = nullptr;

  // Construct sidecar state and compressor chains under the process-level init
  // lock so plugin loading remains serialized.
  pthread_mutex_lock(&cocclInitLock);
  if (cocclCommAvailable(comm)) {
    pthread_mutex_unlock(&cocclInitLock);
    return ncclSuccess;
  }
  NCCLCHECKGOTO(cocclCommCreate(comm), ret, fail_locked);
  cocclTrainingAssistRegister(comm);
  NCCLCHECKGOTO(cocclBufferCommInit(comm), ret, fail_locked);
  NCCLCHECKGOTO(loadCompressorHandlesLocked(), ret, fail_locked);
  NCCLCHECKGOTO(
      loadCompressorsForComm(comm, &profileOptions), ret, fail_locked);
  pthread_mutex_unlock(&cocclInitLock);

  // Profiling must run without the init lock: single-process multi-GPU init
  // workers need to enter the cooperative sampling phase concurrently.
  {
    ncclResult_t profileResult = cocclAutotuneProfile(comm, &profileOptions);
    if (profileResult != ncclSuccess && comm->rank == 0) {
      WARN("COCCL autotune profiling failed with %d; using heuristic selection",
           profileResult);
    }
  }

  // Publish the sidecar last so collectives cannot observe partial chains.
  pthread_mutex_lock(&cocclInitLock);
  NCCLCHECKGOTO(cocclCommCommit(comm), ret, fail_locked);
  pthread_mutex_unlock(&cocclInitLock);
  return ncclSuccess;

fail_locked:
  cocclTrainingAssistUnregister(comm);
  (void)cocclBufferCommDestroy(comm);
  (void)cocclCommDestroy(comm, &registryEmpty, &detachedResources);
  if (registryEmpty) (void)cocclBufferDestroyAll();
  pthread_mutex_unlock(&cocclInitLock);
  (void)cocclCommDestroyDetachedResources(detachedResources);
  return ret;
}

ncclResult_t cocclDestroy(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  bool registryEmpty = false;
  cocclCommDetachedResources* detachedResources = nullptr;

  // Training assist owns an independent raw-comm registry and may have a key
  // even when cocclComm initialization never committed.
  cocclTrainingAssistUnregister(comm);
  pthread_mutex_lock(&cocclInitLock);
  NCCLCHECKGOTO(cocclBufferCommDestroy(comm), ret, fail);
  NCCLCHECKGOTO(
      cocclCommDestroy(comm, &registryEmpty, &detachedResources), ret, fail);
  if (registryEmpty) {
    NCCLCHECKGOTO(cocclBufferDestroyAll(), ret, fail);
  }

exit:
  pthread_mutex_unlock(&cocclInitLock);
  {
    ncclResult_t destroyRet =
        cocclCommDestroyDetachedResources(detachedResources);
    if (ret == ncclSuccess) ret = destroyRet;
  }
  return ret;
fail:
  goto exit;
}
