#include "coccl_buffer_management.h"
#include "coccl_config.h"
#include "comm.h"
#include "debug.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>

namespace {

struct FakeEvent {
  bool ready = false;
};

std::map<void*, size_t> allocations;
std::set<void*> registrations;
std::set<cudaEvent_t> events;
cocclConfig config;
bool failNextRegistration = false;
bool failNextEventRecord = false;
bool completeRecordedEvents = true;
void* lastAllocation = nullptr;

void fail(const char* expression, int line) {
  std::fprintf(stderr, "line %d: %s\n", line, expression);
  std::exit(1);
}

#define EXPECT(expression) \
  do { if (!(expression)) fail(#expression, __LINE__); } while (0)

}  // namespace

thread_local int ncclDebugNoWarn = 0;
int ncclDebugLevel = NCCL_LOG_NONE;
uint64_t ncclDebugMask = 0;
pthread_mutex_t ncclDebugLock = PTHREAD_MUTEX_INITIALIZER;
FILE* ncclDebugFile = nullptr;
char ncclLastError[1024] = {};

void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int,
                  const char*, ...) {}

const cocclConfig& cocclGetConfig() {
  return config;
}

extern "C" ncclResult_t ncclMemAlloc(void** ptr, size_t bytes) {
  *ptr = std::malloc(bytes);
  if (*ptr == nullptr) return ncclSystemError;
  allocations[*ptr] = bytes;
  lastAllocation = *ptr;
  return ncclSuccess;
}

extern "C" ncclResult_t ncclMemFree(void* ptr) {
  allocations.erase(ptr);
  std::free(ptr);
  return ncclSuccess;
}

extern "C" ncclResult_t ncclCommRegister(const ncclComm_t, void*, size_t,
                                           void** handle) {
  if (failNextRegistration) {
    failNextRegistration = false;
    return ncclSystemError;
  }
  *handle = std::malloc(1);
  registrations.insert(*handle);
  return ncclSuccess;
}

extern "C" ncclResult_t ncclCommDeregister(const ncclComm_t, void* handle) {
  registrations.erase(handle);
  std::free(handle);
  return ncclSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaSetDevice(int) {
  return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaDeviceSynchronize() {
  return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaEventCreateWithFlags(
    cudaEvent_t* event, unsigned int) {
  FakeEvent* value = new FakeEvent();
  *event = reinterpret_cast<cudaEvent_t>(value);
  events.insert(*event);
  return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event,
                                                   cudaStream_t) {
  if (failNextEventRecord) {
    failNextEventRecord = false;
    return cudaErrorUnknown;
  }
  reinterpret_cast<FakeEvent*>(event)->ready = completeRecordedEvents;
  return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event) {
  return reinterpret_cast<FakeEvent*>(event)->ready
      ? cudaSuccess : cudaErrorNotReady;
}

extern "C" cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
  events.erase(event);
  delete reinterpret_cast<FakeEvent*>(event);
  return cudaSuccess;
}

extern "C" const char* CUDARTAPI cudaGetErrorString(cudaError_t) {
  return "fake CUDA error";
}

extern "C" cudaError_t CUDARTAPI cudaGetLastError() {
  return cudaSuccess;
}

int main() {
  ncclComm firstComm = {};
  ncclComm secondComm = {};
  firstComm.cudaDev = 0;
  secondComm.cudaDev = 0;
  cudaStream_t firstStream = reinterpret_cast<cudaStream_t>(1);
  cudaStream_t secondStream = reinterpret_cast<cudaStream_t>(2);

  EXPECT(cocclBufferCommInit(&firstComm) == ncclSuccess);
  EXPECT(cocclBufferCommInit(&secondComm) == ncclSuccess);

  cocclBufferHandle first;
  cocclBufferHandle concurrent;
  EXPECT(cocclGetBuffer(&firstComm, 1024, firstStream, &first) ==
         ncclSuccess);
  EXPECT(cocclGetBuffer(&firstComm, 1024, secondStream, &concurrent) ==
         ncclSuccess);
  EXPECT(first.ptr != concurrent.ptr);
  void* reusablePtr = first.ptr;

  completeRecordedEvents = false;
  EXPECT(cocclReleaseBuffer(&first, firstStream) == ncclSuccess);
  cocclBufferHandle crossStream;
  EXPECT(cocclGetBuffer(&firstComm, 512, secondStream, &crossStream) ==
         ncclSuccess);
  EXPECT(crossStream.ptr != reusablePtr);
  cocclBufferHandle smaller;
  EXPECT(cocclGetBuffer(&firstComm, 512, firstStream, &smaller) ==
         ncclSuccess);
  EXPECT(smaller.ptr == reusablePtr);
  completeRecordedEvents = true;
  EXPECT(cocclReleaseBuffer(&crossStream, secondStream) == ncclSuccess);
  EXPECT(cocclReleaseBuffer(&smaller, firstStream) == ncclSuccess);
  EXPECT(cocclReleaseBuffer(&concurrent, secondStream) == ncclSuccess);

  cocclBufferHandle independent;
  EXPECT(cocclGetBuffer(&secondComm, 1024, firstStream, &independent) ==
         ncclSuccess);
  EXPECT(independent.ptr != reusablePtr);
  EXPECT(cocclReleaseBuffer(&independent, firstStream) == ncclSuccess);
  const size_t allocationsBeforeFirstDestroy = allocations.size();
  EXPECT(cocclBufferCommDestroy(&firstComm) == ncclSuccess);
  EXPECT(allocations.size() < allocationsBeforeFirstDestroy);
  EXPECT(!allocations.empty());
  EXPECT(cocclBufferCommDestroy(&secondComm) == ncclSuccess);
  EXPECT(allocations.empty() && registrations.empty() && events.empty());

  EXPECT(cocclBufferCommInit(&firstComm) == ncclSuccess);
  failNextRegistration = true;
  cocclBufferHandle failed;
  EXPECT(cocclGetBuffer(&firstComm, 2048, firstStream, &failed) ==
         ncclSystemError);
  EXPECT(failed.ptr == nullptr && failed.slice == nullptr);
  void* rolledBackAllocation = lastAllocation;

  cocclBufferHandle recovered;
  EXPECT(cocclGetBuffer(&firstComm, 2048, firstStream, &recovered) ==
         ncclSuccess);
  EXPECT(recovered.ptr == rolledBackAllocation);

  failNextEventRecord = true;
  EXPECT(cocclReleaseBuffer(&recovered, firstStream) ==
         ncclUnhandledCudaError);
  EXPECT(recovered.slice != nullptr);
  EXPECT(cocclReleaseBuffer(&recovered, firstStream) == ncclSuccess);

  cocclBufferHandle afterError;
  EXPECT(cocclGetBuffer(&firstComm, 2048, firstStream, &afterError) ==
         ncclSuccess);
  EXPECT(afterError.ptr == rolledBackAllocation);
  EXPECT(cocclReleaseBuffer(&afterError, firstStream) == ncclSuccess);
  EXPECT(cocclBufferCommDestroy(&firstComm) == ncclSuccess);
  EXPECT(allocations.empty() && registrations.empty() && events.empty());

  std::puts("M3 buffer manager tests passed");
  return 0;
}
