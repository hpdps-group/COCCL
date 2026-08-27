#include "core/memory/coccl_buffer_management.h"
#include "core/config/coccl_config.h"
#include "comm.h"
#include "cudawrap.h"
#include "debug.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

constexpr size_t kMiB = 1024 * 1024;

struct FakeEvent {
  bool ready = false;
};

cocclConfig config;
bool vmmEnabled = true;
bool completeRecordedEvents = true;
bool failNextEventRecord = false;
uint64_t nextPhysicalHandle = 1;
std::map<void*, size_t> legacyAllocations;
std::map<CUdeviceptr, size_t> reservations;
std::map<CUmemGenericAllocationHandle, size_t> physicalHandles;
std::set<void*> registrations;
std::set<cudaEvent_t> events;
std::vector<std::string> lifecycle;

struct RegistrationCall {
  ncclComm_t comm;
  void* ptr;
  size_t bytes;
};

std::vector<RegistrationCall> registrationCalls;

void fail(const char* expression, int line) {
  std::fprintf(stderr, "line %d: %s\n", line, expression);
  std::exit(1);
}

#define EXPECT(expression) \
  do { if (!(expression)) fail(#expression, __LINE__); } while (0)

size_t operationIndex(const char* operation) {
  auto found = std::find(lifecycle.begin(), lifecycle.end(), operation);
  EXPECT(found != lifecycle.end());
  return static_cast<size_t>(found - lifecycle.begin());
}

CUresult CUDAAPI fakeCuDeviceGet(CUdevice* device, int ordinal) {
  *device = ordinal;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI fakeCuDeviceGetAttribute(int* value, CUdevice_attribute,
                                          CUdevice) {
  *value = 1;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI fakeCuGetErrorString(CUresult, const char** text) {
  *text = "fake CUDA driver error";
  return CUDA_SUCCESS;
}

CUresult CUDAAPI fakeCuMemAddressReserve(CUdeviceptr* ptr, size_t bytes,
                                         size_t, CUdeviceptr,
                                         unsigned long long) {
  void* allocation = std::malloc(bytes);
  if (allocation == nullptr) return CUDA_ERROR_OUT_OF_MEMORY;
  *ptr = reinterpret_cast<CUdeviceptr>(allocation);
  reservations[*ptr] = bytes;
  lifecycle.emplace_back("reserve");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI fakeCuMemAddressFree(CUdeviceptr ptr, size_t) {
  auto found = reservations.find(ptr);
  if (found == reservations.end()) return CUDA_ERROR_INVALID_VALUE;
  std::free(reinterpret_cast<void*>(ptr));
  reservations.erase(found);
  lifecycle.emplace_back("address-free");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI fakeCuMemCreate(CUmemGenericAllocationHandle* handle,
                                 size_t bytes, const CUmemAllocationProp*,
                                 unsigned long long) {
  *handle = nextPhysicalHandle++;
  physicalHandles.emplace(*handle, bytes);
  lifecycle.emplace_back("create");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI fakeCuMemGetAllocationGranularity(
    size_t* granularity, const CUmemAllocationProp*,
    CUmemAllocationGranularity_flags) {
  *granularity = 2 * kMiB;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI fakeCuMemMap(CUdeviceptr, size_t, size_t,
                              CUmemGenericAllocationHandle,
                              unsigned long long) {
  lifecycle.emplace_back("map");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI fakeCuMemRelease(CUmemGenericAllocationHandle handle) {
  physicalHandles.erase(handle);
  lifecycle.emplace_back("release");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI fakeCuMemSetAccess(CUdeviceptr, size_t,
                                    const CUmemAccessDesc*, size_t) {
  lifecycle.emplace_back("access");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI fakeCuMemUnmap(CUdeviceptr, size_t) {
  lifecycle.emplace_back("unmap");
  return CUDA_SUCCESS;
}

}  // namespace

thread_local int ncclDebugNoWarn = 0;
int ncclDebugLevel = NCCL_LOG_NONE;
uint64_t ncclDebugMask = 0;
pthread_mutex_t ncclDebugLock = PTHREAD_MUTEX_INITIALIZER;
FILE* ncclDebugFile = nullptr;
char ncclLastError[1024] = {};

void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int,
                  const char*, ...) {}

CUmemAllocationHandleType ncclCuMemHandleType =
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
PFN_cuDeviceGet pfn_cuDeviceGet = fakeCuDeviceGet;
PFN_cuDeviceGetAttribute pfn_cuDeviceGetAttribute =
    fakeCuDeviceGetAttribute;
PFN_cuGetErrorString pfn_cuGetErrorString = fakeCuGetErrorString;
PFN_cuMemAddressReserve pfn_cuMemAddressReserve =
    fakeCuMemAddressReserve;
PFN_cuMemAddressFree pfn_cuMemAddressFree = fakeCuMemAddressFree;
PFN_cuMemCreate pfn_cuMemCreate = fakeCuMemCreate;
PFN_cuMemGetAllocationGranularity pfn_cuMemGetAllocationGranularity =
    fakeCuMemGetAllocationGranularity;
PFN_cuMemMap pfn_cuMemMap = fakeCuMemMap;
PFN_cuMemRelease pfn_cuMemRelease = fakeCuMemRelease;
PFN_cuMemSetAccess pfn_cuMemSetAccess = fakeCuMemSetAccess;
PFN_cuMemUnmap pfn_cuMemUnmap = fakeCuMemUnmap;

ncclResult_t ncclCudaLibraryInit() {
  return ncclSuccess;
}

int ncclCuMemEnable() {
  return vmmEnabled ? 1 : 0;
}

const cocclConfig& cocclGetConfig() {
  return config;
}

bool cocclConfigInitialize() {
  return true;
}

extern "C" ncclResult_t ncclMemAlloc(void** ptr, size_t bytes) {
  *ptr = std::malloc(bytes);
  if (*ptr == nullptr) return ncclSystemError;
  legacyAllocations[*ptr] = bytes;
  return ncclSuccess;
}

extern "C" ncclResult_t ncclMemFree(void* ptr) {
  legacyAllocations.erase(ptr);
  std::free(ptr);
  return ncclSuccess;
}

extern "C" ncclResult_t ncclCommRegister(ncclComm_t comm, void* ptr,
                                           size_t bytes,
                                           void** handle) {
  *handle = std::malloc(1);
  registrations.insert(*handle);
  registrationCalls.push_back({comm, ptr, bytes});
  lifecycle.emplace_back("register");
  return ncclSuccess;
}

extern "C" ncclResult_t ncclCommDeregister(ncclComm_t, void* handle) {
  registrations.erase(handle);
  std::free(handle);
  lifecycle.emplace_back("deregister");
  return ncclSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaSetDevice(int) {
  return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaGetDeviceCount(int* count) {
  *count = 1;
  return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(
    int* canAccess, int, int) {
  *canAccess = 1;
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

extern "C" cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) {
  reinterpret_cast<FakeEvent*>(event)->ready = true;
  return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
  events.erase(event);
  delete reinterpret_cast<FakeEvent*>(event);
  return cudaSuccess;
}

extern "C" const char* CUDARTAPI cudaGetErrorString(cudaError_t) {
  return "fake CUDA runtime error";
}

extern "C" cudaError_t CUDARTAPI cudaGetLastError() {
  return cudaSuccess;
}

int main() {
  config.buffer.physicalChunkBytes = 8 * kMiB;
  ncclComm firstComm = {};
  ncclComm secondComm = {};
  firstComm.cudaDev = 0;
  secondComm.cudaDev = 0;
  cudaStream_t firstStream = reinterpret_cast<cudaStream_t>(1);

  EXPECT(cocclBufferCommInit(&firstComm) == ncclSuccess);
  EXPECT(cocclBufferCommInit(&secondComm) == ncclSuccess);
  cocclBufferHandle first;
  EXPECT(cocclGetBuffer(&firstComm, 10 * kMiB, firstStream, &first) ==
         ncclSuccess);
  EXPECT(first.bytes == 10 * kMiB);
  const size_t vmmLogicalBytes = first.bytes;
  void* firstAddress = first.ptr;
  EXPECT(physicalHandles.size() == 1 &&
         physicalHandles.begin()->second == 16 * kMiB &&
         reservations.size() == 1);
  EXPECT(registrationCalls.size() == 1);
  EXPECT(registrationCalls[0].comm == &firstComm &&
         registrationCalls[0].ptr == firstAddress &&
         registrationCalls[0].bytes == 16 * kMiB);
  EXPECT(cocclRegisterBufferForComm(&first, &secondComm) == ncclSuccess);
  EXPECT(registrations.size() == 2 && registrationCalls.size() == 2);
  EXPECT(registrationCalls[1].comm == &secondComm &&
         registrationCalls[1].ptr == firstAddress &&
         registrationCalls[1].bytes == 16 * kMiB);
  EXPECT(cocclRegisterBufferForComm(&first, &secondComm) == ncclSuccess);
  EXPECT(registrations.size() == 2 && registrationCalls.size() == 2);

  completeRecordedEvents = false;
  EXPECT(cocclReleaseBuffer(&first, firstStream) == ncclSuccess);
  cocclBufferHandle smaller;
  EXPECT(cocclGetBuffer(&firstComm, 4 * kMiB, firstStream, &smaller) ==
         ncclSuccess);
  EXPECT(smaller.ptr == firstAddress && physicalHandles.size() == 1);

  completeRecordedEvents = true;
  EXPECT(cocclReleaseBuffer(&smaller, firstStream) == ncclSuccess);
  cocclBufferHandle grown;
  EXPECT(cocclGetBuffer(&firstComm, 20 * kMiB, firstStream, &grown) ==
         ncclSuccess);
  EXPECT(grown.bytes == 20 * kMiB);
  EXPECT(physicalHandles.size() == 1 &&
         physicalHandles.begin()->second == 24 * kMiB &&
         reservations.size() == 1);
  EXPECT(cocclRegisterBufferForComm(&grown, &secondComm) == ncclSuccess);
  EXPECT(registrations.size() == 2);
  failNextEventRecord = true;
  EXPECT(cocclReleaseBuffer(&grown, firstStream) ==
         ncclUnhandledCudaError);
  EXPECT(grown.slice != nullptr);
  EXPECT(cocclReleaseBuffer(&grown, firstStream) == ncclSuccess);

  cocclBufferHandle independent;
  EXPECT(cocclGetBuffer(&secondComm, 8 * kMiB, firstStream, &independent) ==
         ncclSuccess);
  EXPECT(physicalHandles.size() == 2 && reservations.size() == 2);
  EXPECT(cocclReleaseBuffer(&independent, firstStream) == ncclSuccess);
  EXPECT(registrations.size() == 3);

  lifecycle.clear();
  EXPECT(cocclBufferCommDestroy(&secondComm) == ncclSuccess);
  EXPECT(physicalHandles.size() == 1 && reservations.size() == 1);
  EXPECT(registrations.size() == 1);
  EXPECT(operationIndex("deregister") < operationIndex("unmap"));
  EXPECT(operationIndex("unmap") < operationIndex("release"));
  EXPECT(operationIndex("release") < operationIndex("address-free"));
  lifecycle.clear();
  EXPECT(cocclBufferCommDestroy(&firstComm) == ncclSuccess);
  EXPECT(physicalHandles.empty() && reservations.empty() &&
         registrations.empty() && events.empty());

  vmmEnabled = false;
  EXPECT(cocclBufferCommInit(&firstComm) == ncclSuccess);
  cocclBufferHandle legacy;
  EXPECT(cocclGetBuffer(&firstComm, 10 * kMiB, firstStream, &legacy) ==
         ncclSuccess);
  EXPECT(legacy.bytes == vmmLogicalBytes && legacyAllocations.size() == 1);
  EXPECT(cocclReleaseBuffer(&legacy, firstStream) == ncclSuccess);
  EXPECT(cocclBufferCommDestroy(&firstComm) == ncclSuccess);
  EXPECT(legacyAllocations.empty() && registrations.empty() && events.empty());

  std::puts("VMM buffer tests passed");
  return 0;
}
