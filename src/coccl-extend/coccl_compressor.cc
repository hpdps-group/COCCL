#include "compression/coccl_compressor.h"

#include "buffer/coccl_buffer_management.h"
#include "comm.h"
#include "debug.h"

#include <array>
#include <new>
#include <vector>

namespace {

constexpr size_t kInlineScratchBuffers = 4;
constexpr size_t kInlinePersistentSlots = 4;

struct cocclCompressorCompletion {
  cudaStream_t stream = nullptr;
  cudaEvent_t event = nullptr;
};

struct cocclCompressorPersistentSlot {
  cocclBufferHandle buffer = {};
  cudaStream_t lastStream = nullptr;
  cudaEvent_t completion = nullptr;
};

}  // namespace

struct cocclCompressorRuntimeState {
  ncclComm_t ownerComm = nullptr;
  const cocclCompressorPlugin* compressor = nullptr;
  void* config = nullptr;
  bool ownsConfig = false;

  void* instance = nullptr;
  const void* instanceTypeKey = nullptr;
  cocclCompressorDestroyStateFn destroyInstance = nullptr;
  std::vector<cocclCompressorCompletion> instanceCompletions;
  std::vector<cocclCompressorPersistentSlot> persistentSlots;

  ~cocclCompressorRuntimeState() {
    if (ownerComm != nullptr) (void)cudaSetDevice(ownerComm->cudaDev);

    for (const auto& completion : instanceCompletions) {
      if (completion.event != nullptr) {
        (void)cudaEventSynchronize(completion.event);
      }
    }
    for (const auto& slot : persistentSlots) {
      if (slot.completion != nullptr) {
        (void)cudaEventSynchronize(slot.completion);
      }
    }

    if (destroyInstance != nullptr && instance != nullptr) {
      destroyInstance(instance);
    }
    instance = nullptr;

    cudaStream_t cleanupStream = nullptr;
    bool ownsCleanupStream = false;
    for (const auto& slot : persistentSlots) {
      if (slot.buffer.ptr != nullptr) {
        ownsCleanupStream =
            cudaStreamCreateWithFlags(&cleanupStream,
                                      cudaStreamNonBlocking) == cudaSuccess;
        break;
      }
    }
    for (auto& slot : persistentSlots) {
      if (slot.buffer.ptr != nullptr) {
        (void)cocclReleaseBuffer(&slot.buffer, cleanupStream);
      }
      if (slot.completion != nullptr) {
        (void)cudaEventDestroy(slot.completion);
      }
    }
    if (cleanupStream != nullptr) {
      (void)cudaStreamSynchronize(cleanupStream);
      if (ownsCleanupStream) (void)cudaStreamDestroy(cleanupStream);
    }
    persistentSlots.clear();

    for (const auto& completion : instanceCompletions) {
      if (completion.event != nullptr) {
        (void)cudaEventDestroy(completion.event);
      }
    }
    instanceCompletions.clear();

    if (ownsConfig && compressor != nullptr &&
        compressor->destroyConfig != nullptr && config != nullptr) {
      compressor->destroyConfig(config);
    }
    config = nullptr;
  }
};

namespace {

struct cocclCompressorInvocation {
  cocclCompressorRuntimeState* state = nullptr;
  cudaStream_t stream = nullptr;
  std::array<cocclBufferHandle, kInlineScratchBuffers> scratchBuffers = {};
  size_t scratchBufferCount = 0;
  std::vector<cocclBufferHandle> overflowScratchBuffers;
  std::array<size_t, kInlinePersistentSlots> touchedPersistentSlots = {};
  size_t touchedPersistentCount = 0;
  std::vector<size_t> overflowTouchedPersistentSlots;
  bool usedInstance = false;
};

bool checkedMultiply(size_t lhs, size_t rhs, size_t* result) {
  if (result == nullptr || (lhs != 0 && rhs > SIZE_MAX / lhs)) return false;
  *result = lhs * rhs;
  return true;
}

size_t typeSize(ncclDataType_t datatype) {
  const int bytes = ncclTypeSize(datatype);
  return bytes > 0 ? (size_t)bytes : 0;
}

uint64_t capabilityForOperation(cocclCompressorOperation operation) {
  switch (operation) {
    case cocclCompressorOperationCompress:
      return cocclCompressorCapabilityCompress;
    case cocclCompressorOperationDecompress:
      return cocclCompressorCapabilityDecompress;
    case cocclCompressorOperationDecompressReduce:
      return cocclCompressorCapabilityDecompressReduce;
    case cocclCompressorOperationDecompressReduceCompress:
      return cocclCompressorCapabilityDecompressReduceCompress;
  }
  return 0;
}

ncclResult_t allocateScratch(void* context, size_t bytes,
                             cocclCompressorBufferView* output) {
  auto* invocation = static_cast<cocclCompressorInvocation*>(context);
  if (invocation == nullptr || invocation->state == nullptr ||
      invocation->state->ownerComm == nullptr || output == nullptr ||
      bytes == 0) {
    return ncclInvalidArgument;
  }

  cocclBufferHandle buffer = {};
  NCCLCHECK(cocclGetUnregisteredBuffer(invocation->state->ownerComm, bytes,
                                       &buffer));
  if (invocation->scratchBufferCount < invocation->scratchBuffers.size()) {
    invocation->scratchBuffers[invocation->scratchBufferCount++] = buffer;
  } else {
    try {
      invocation->overflowScratchBuffers.push_back(buffer);
    } catch (const std::bad_alloc&) {
      (void)cocclReleaseBuffer(&buffer, invocation->stream);
      return ncclSystemError;
    }
  }
  *output = {buffer.ptr, buffer.bytes};
  return ncclSuccess;
}

bool persistentSlotTouched(const cocclCompressorInvocation* invocation,
                           size_t slot) {
  for (size_t i = 0; i < invocation->touchedPersistentCount; ++i) {
    if (invocation->touchedPersistentSlots[i] == slot) return true;
  }
  for (size_t touched : invocation->overflowTouchedPersistentSlots) {
    if (touched == slot) return true;
  }
  return false;
}

ncclResult_t rememberPersistentSlot(cocclCompressorInvocation* invocation,
                                    size_t slot) {
  if (persistentSlotTouched(invocation, slot)) return ncclSuccess;
  if (invocation->touchedPersistentCount <
      invocation->touchedPersistentSlots.size()) {
    invocation->touchedPersistentSlots[
        invocation->touchedPersistentCount++] = slot;
    return ncclSuccess;
  }
  try {
    invocation->overflowTouchedPersistentSlots.push_back(slot);
  } catch (const std::bad_alloc&) {
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t acquirePersistent(void* context, size_t slot, size_t bytes,
                               cocclCompressorBufferView* output) {
  auto* invocation = static_cast<cocclCompressorInvocation*>(context);
  if (invocation == nullptr || invocation->state == nullptr ||
      invocation->state->ownerComm == nullptr || output == nullptr ||
      bytes == 0) {
    return ncclInvalidArgument;
  }

  auto* state = invocation->state;
  if (slot >= state->persistentSlots.max_size()) return ncclInvalidArgument;
  try {
    if (state->persistentSlots.size() <= slot) {
      state->persistentSlots.resize(slot + 1);
    }
  } catch (const std::bad_alloc&) {
    return ncclSystemError;
  }

  cocclCompressorPersistentSlot& persistent = state->persistentSlots[slot];
  if (persistent.completion != nullptr &&
      persistent.lastStream != invocation->stream) {
    CUDACHECK(cudaStreamWaitEvent(invocation->stream,
                                  persistent.completion, 0));
  }
  if (persistent.buffer.ptr == nullptr || persistent.buffer.bytes < bytes) {
    if (persistent.buffer.ptr != nullptr) {
      NCCLCHECK(cocclReleaseBuffer(&persistent.buffer, invocation->stream));
    }
    NCCLCHECK(cocclGetUnregisteredBuffer(state->ownerComm, bytes,
                                         &persistent.buffer));
  }
  NCCLCHECK(rememberPersistentSlot(invocation, slot));
  *output = {persistent.buffer.ptr, persistent.buffer.bytes};
  return ncclSuccess;
}

ncclResult_t getOrCreateState(void* context, const void* typeKey,
                              cocclCompressorCreateStateFn createState,
                              cocclCompressorDestroyStateFn destroyState,
                              void** output) {
  auto* invocation = static_cast<cocclCompressorInvocation*>(context);
  if (invocation == nullptr || invocation->state == nullptr ||
      typeKey == nullptr || createState == nullptr || destroyState == nullptr ||
      output == nullptr) {
    return ncclInvalidArgument;
  }
  auto* state = invocation->state;
  if (state->instance != nullptr) {
    if (state->instanceTypeKey != typeKey ||
        state->destroyInstance != destroyState) {
      return ncclInvalidUsage;
    }
  } else {
    CUDACHECK(cudaSetDevice(state->ownerComm->cudaDev));
    void* created = nullptr;
    NCCLCHECK(createState(&created));
    if (created == nullptr) return ncclInvalidUsage;
    state->instance = created;
    state->instanceTypeKey = typeKey;
    state->destroyInstance = destroyState;
  }
  invocation->usedInstance = true;
  *output = state->instance;
  return ncclSuccess;
}

const cocclCompressorHostApi executionHostApi = {
    COCCL_COMPRESSOR_HOST_API_VERSION,
    sizeof(cocclCompressorHostApi),
    allocateScratch,
    acquirePersistent,
    getOrCreateState,
};

ncclResult_t recordCompletion(cudaStream_t stream,
                              std::vector<cocclCompressorCompletion>* events) {
  cudaEvent_t event = nullptr;
  for (const auto& completion : *events) {
    if (completion.stream == stream) {
      event = completion.event;
      break;
    }
  }
  if (event == nullptr) {
    CUDACHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    try {
      events->push_back({stream, event});
    } catch (const std::bad_alloc&) {
      (void)cudaEventDestroy(event);
      return ncclSystemError;
    }
  }
  CUDACHECK(cudaEventRecord(event, stream));
  return ncclSuccess;
}

ncclResult_t finishInvocation(cocclCompressorInvocation* invocation,
                              ncclResult_t callbackResult) {
  if (invocation == nullptr || invocation->state == nullptr) {
    return callbackResult == ncclSuccess ? ncclInvalidArgument
                                         : callbackResult;
  }
  ncclResult_t ret = callbackResult;
  auto releaseScratch = [&](cocclBufferHandle* buffer) {
    ncclResult_t releaseResult =
        cocclReleaseBuffer(buffer, invocation->stream);
    if (ret == ncclSuccess) ret = releaseResult;
  };
  for (size_t i = 0; i < invocation->scratchBufferCount; ++i) {
    releaseScratch(&invocation->scratchBuffers[i]);
  }
  for (auto& buffer : invocation->overflowScratchBuffers) {
    releaseScratch(&buffer);
  }

  auto recordPersistent = [&](size_t slotIndex) {
    cocclCompressorPersistentSlot& slot =
        invocation->state->persistentSlots[slotIndex];
    if (slot.completion == nullptr) {
      cudaError_t result = cudaEventCreateWithFlags(
          &slot.completion, cudaEventDisableTiming);
      if (result != cudaSuccess && ret == ncclSuccess) {
        ret = ncclUnhandledCudaError;
      }
    }
    if (slot.completion != nullptr) {
      cudaError_t result = cudaEventRecord(slot.completion,
                                           invocation->stream);
      if (result != cudaSuccess && ret == ncclSuccess) {
        ret = ncclUnhandledCudaError;
      } else if (result == cudaSuccess) {
        slot.lastStream = invocation->stream;
      }
    }
  };
  for (size_t i = 0; i < invocation->touchedPersistentCount; ++i) {
    recordPersistent(invocation->touchedPersistentSlots[i]);
  }
  for (size_t slotIndex : invocation->overflowTouchedPersistentSlots) {
    recordPersistent(slotIndex);
  }
  if (invocation->usedInstance) {
    ncclResult_t result = recordCompletion(
        invocation->stream, &invocation->state->instanceCompletions);
    if (ret == ncclSuccess) ret = result;
  }
  return ret;
}

ncclResult_t validateExecution(const cocclCompressorHandle& handle,
                               cocclCompressorOperation operation,
                               const cocclCompressorDataView& input,
                               cocclCompressorOutputView* output,
                               cocclCompressorRuntimeState** state) {
  if (!handle || output == nullptr || state == nullptr ||
      input.data == nullptr || output->data == nullptr ||
      input.chunks == 0 || input.elements % input.chunks != 0) {
    return ncclInvalidArgument;
  }
  const size_t inputTypeBytes = typeSize(input.datatype);
  size_t expectedInputBytes = 0;
  if (inputTypeBytes == 0 ||
      !checkedMultiply(input.elements, inputTypeBytes,
                       &expectedInputBytes) ||
      expectedInputBytes != input.bytes) {
    return ncclInvalidArgument;
  }
  *state = handle.state.get();
  const uint64_t capability = capabilityForOperation(operation);
  if ((*state)->compressor == nullptr || capability == 0 ||
      (((*state)->compressor->capabilities & capability) == 0)) {
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

ncclResult_t validateOutput(void* expectedData,
                            const cocclCompressorOutputView* output) {
  if (output == nullptr || output->data != expectedData ||
      output->bytes > output->capacityBytes || output->chunks == 0 ||
      output->elements % output->chunks != 0) {
    return ncclInvalidArgument;
  }
  size_t expectedBytes = 0;
  const size_t outputTypeBytes = typeSize(output->datatype);
  if (outputTypeBytes == 0 ||
      !checkedMultiply(output->elements, outputTypeBytes, &expectedBytes) ||
      expectedBytes != output->bytes) {
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

}  // namespace

ncclResult_t cocclCreateCompressorHandle(
    ncclComm_t comm, const cocclCompressorPlugin* compressor,
    void* parsedConfig, cocclCompressorHandle* handle) {
  if (comm == nullptr || compressor == nullptr || handle == nullptr) {
    return ncclInvalidArgument;
  }
  *handle = {};
  std::shared_ptr<cocclCompressorRuntimeState> state(
      new (std::nothrow) cocclCompressorRuntimeState());
  if (!state) return ncclSystemError;
  state->ownerComm = comm;
  state->compressor = compressor;
  state->config = parsedConfig;
  state->ownsConfig = true;
  handle->state = std::move(state);
  return ncclSuccess;
}

const cocclCompressorPlugin* cocclCompressorDescriptor(
    const cocclCompressorHandle& handle) {
  return handle ? handle.state->compressor : nullptr;
}

bool cocclCompressorSupports(const cocclCompressorHandle& handle,
                             cocclCompressorCapability capability) {
  const cocclCompressorPlugin* plugin = cocclCompressorDescriptor(handle);
  return plugin != nullptr && (plugin->capabilities & capability) != 0;
}

ncclResult_t cocclExecuteCompressor(
    const cocclCompressorHandle& handle,
    cocclCompressorOperation operation,
    const cocclCompressorDataView& input, cocclCompressorOutputView* output,
    int rank, size_t reduceChunks, ncclDataType_t originalDatatype,
    size_t originalElements, cudaStream_t stream) {
  cocclCompressorRuntimeState* state = nullptr;
  NCCLCHECK(validateExecution(handle, operation, input, output, &state));
  if ((operation == cocclCompressorOperationDecompressReduce ||
       operation ==
           cocclCompressorOperationDecompressReduceCompress) &&
      reduceChunks == 0) {
    return ncclInvalidArgument;
  }

  void* expectedData = output->data;
  cocclCompressorInvocation invocation = {state, stream};
  cocclCompressorExecutionContext execution = {
      sizeof(cocclCompressorExecutionContext),
      &executionHostApi,
      &invocation,
      stream,
      state->ownerComm->cudaDev,
      state->ownerComm->rank,
      state->ownerComm->nRanks,
      state->ownerComm->nNodes,
      state->ownerComm->localRanks,
  };
  cocclCompressorCall call = {
      sizeof(cocclCompressorCall), operation, input, output, rank,
      reduceChunks, originalDatatype, originalElements, state->config,
      &execution,
  };
  ncclResult_t ret = finishInvocation(
      &invocation, state->compressor->execute(&call));
  return ret == ncclSuccess ? validateOutput(expectedData, output) : ret;
}
