/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef MULTIBYTE_ANS_INCLUDE_UTILS_DEVICEUTILS_H
#define MULTIBYTE_ANS_INCLUDE_UTILS_DEVICEUTILS_H

#pragma once
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

// #include "DeviceUtils0.h"
#include <cuda_profiler_api.h>
#include <mutex>
#include <unordered_map>

/// Wrapper to test return status of CUDA functions
#define CUDA_VERIFY(X)                                         \
  do {                                                         \
    auto err__ = (X);                                          \
    if (err__ != cudaSuccess) {\
    std::cout << "CUDA error " << multibyte_ans::errorToName(err__) << " "\
              << multibyte_ans::errorToString(err__) << std::endl;\
    }\
  } while (0)

/// Wrapper to synchronously probe for CUDA errors
// #define GPU_SYNC_ERROR 1

#ifdef GPU_SYNC_ERROR
#define CUDA_TEST_ERROR()                 \
  do {                                    \
    CUDA_VERIFY(cudaDeviceSynchronize()); \
  } while (0)
#else
#define CUDA_TEST_ERROR()            \
  do {                               \
    CUDA_VERIFY(cudaGetLastError()); \
  } while (0)
#endif

namespace multibyte_ans {
  
constexpr int kWarpSize = 32;

/// std::string wrapper around cudaGetErrorString
std::string errorToString(cudaError_t err);

/// std::string wrapper around cudaGetErrorName
std::string errorToName(cudaError_t err);

/// Returns the current thread-local GPU device
int getCurrentDevice();

/// Sets the current thread-local GPU device
void setCurrentDevice(int device);

/// Returns the number of available GPU devices
int getNumDevices();

/// Starts the CUDA profiler (exposed via SWIG)
void profilerStart();

/// Stops the CUDA profiler (exposed via SWIG)
void profilerStop();

/// Synchronizes the CPU against all devices (equivalent to
/// cudaDeviceSynchronize for each device)
void synchronizeAllDevices();

/// Returns a cached cudaDeviceProp for the given device
const cudaDeviceProp& getDeviceProperties(int device);

/// Returns the cached cudaDeviceProp for the current device
const cudaDeviceProp& getCurrentDeviceProperties();

/// Returns the maximum number of threads available for the given GPU
/// device
int getMaxThreads(int device);

/// Equivalent to getMaxThreads(getCurrentDevice())
int getMaxThreadsCurrentDevice();

/// Returns the maximum smem available for the given GPU device
size_t getMaxSharedMemPerBlock(int device);

/// Equivalent to getMaxSharedMemPerBlock(getCurrentDevice())
size_t getMaxSharedMemPerBlockCurrentDevice();

/// For a given pointer, returns whether or not it is located on
/// a device (deviceId >= 0) or the host (-1).
int getDeviceForAddress(const void* p);

/// Does the given device support full unified memory sharing host
/// memory?
bool getFullUnifiedMemSupport(int device);

/// Equivalent to getFullUnifiedMemSupport(getCurrentDevice())
bool getFullUnifiedMemSupportCurrentDevice();

/// RAII object to set the current device, and restore the previous
/// device upon destruction
class DeviceScope {
 public:
  explicit DeviceScope(int device);
  ~DeviceScope();

 private:
  int prevDevice_;
};

// RAII object to manage a cudaEvent_t
class CudaEvent {
 public:
  /// Creates an event and records it in this stream
  explicit CudaEvent(cudaStream_t stream, bool timer = false);
  CudaEvent(const CudaEvent& event) = delete;
  CudaEvent(CudaEvent&& event) noexcept;
  ~CudaEvent();

  CudaEvent& operator=(CudaEvent&& event) noexcept;
  CudaEvent& operator=(CudaEvent& event) = delete;

  inline cudaEvent_t get() {
    return event_;
  }

  /// Wait on this event in this stream
  void streamWaitOnEvent(cudaStream_t stream);

  /// Have the CPU wait for the completion of this event
  void cpuWaitOnEvent();

  /// Returns the elapsed time from the other event
  float timeFrom(CudaEvent& from);

 private:
  cudaEvent_t event_;
};

// RAII object to manage a cudaStream_t
class CudaStream {
 public:
  /// Creates a stream on the current device
  CudaStream(int flags = cudaStreamDefault);
  CudaStream(const CudaStream& stream) = delete;
  CudaStream(CudaStream&& stream) noexcept;
  ~CudaStream();

  CudaStream& operator=(CudaStream&& stream) noexcept;
  CudaStream& operator=(CudaStream& stream) = delete;

  inline cudaStream_t get() {
    return stream_;
  }

  operator cudaStream_t() {
    return stream_;
  }

  static CudaStream make();
  static CudaStream makeNonBlocking();

 private:
  cudaStream_t stream_;
};

/// Call for a collection of streams to wait on
template <typename L1, typename L2>
void streamWaitBase(const L1& listWaiting, const L2& listWaitOn) {
  // For all the streams we are waiting on, create an event
  std::vector<cudaEvent_t> events;
  for (auto& stream : listWaitOn) {
    cudaEvent_t event;
    CUDA_VERIFY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    CUDA_VERIFY(cudaEventRecord(event, stream));
    events.push_back(event);
  }

  // For all the streams that are waiting, issue a wait
  for (auto& stream : listWaiting) {
    for (auto& event : events) {
      CUDA_VERIFY(cudaStreamWaitEvent(stream, event, 0));
    }
  }

  for (auto& event : events) {
    CUDA_VERIFY(cudaEventDestroy(event));
  }
}

/// These versions allow usage of initializer_list as arguments, since
/// otherwise {...} doesn't have a type
template <typename L1>
void streamWait(const L1& a, const std::initializer_list<cudaStream_t>& b) {
  streamWaitBase(a, b);
}

template <typename L2>
void streamWait(const std::initializer_list<cudaStream_t>& a, const L2& b) {
  streamWaitBase(a, b);
}

inline void streamWait(
    const std::initializer_list<cudaStream_t>& a,
    const std::initializer_list<cudaStream_t>& b) {
  streamWaitBase(a, b);
}

std::string errorToString(cudaError_t err) {
  return std::string(cudaGetErrorString(err));
}

std::string errorToName(cudaError_t err) {
  return std::string(cudaGetErrorName(err));
}

int getCurrentDevice() {
  int dev = -1;
  CUDA_VERIFY(cudaGetDevice(&dev));

  return dev;
}

void setCurrentDevice(int device) {
  CUDA_VERIFY(cudaSetDevice(device));
}

int getNumDevices() {
  int numDev = -1;
  cudaError_t err = cudaGetDeviceCount(&numDev);
  if (cudaErrorNoDevice == err) {
    numDev = 0;
  } else {
    CUDA_VERIFY(err);
  }

  return numDev;
}

void profilerStart() {
  CUDA_VERIFY(cudaProfilerStart());
}

void profilerStop() {
  CUDA_VERIFY(cudaProfilerStop());
}

void synchronizeAllDevices() {
  for (int i = 0; i < getNumDevices(); ++i) {
    DeviceScope scope(i);

    CUDA_VERIFY(cudaDeviceSynchronize());
  }
}

const cudaDeviceProp& getDeviceProperties(int device) {
  static std::mutex mutex;
  static std::unordered_map<int, cudaDeviceProp> properties;

  std::lock_guard<std::mutex> guard(mutex);

  auto it = properties.find(device);
  if (it == properties.end()) {
    cudaDeviceProp prop;
    CUDA_VERIFY(cudaGetDeviceProperties(&prop, device));

    properties[device] = prop;
    it = properties.find(device);
  }

  return it->second;
}

const cudaDeviceProp& getCurrentDeviceProperties() {
  return getDeviceProperties(getCurrentDevice());
}

int getMaxThreads(int device) {
  return getDeviceProperties(device).maxThreadsPerBlock;
}

int getMaxThreadsCurrentDevice() {
  return getMaxThreads(getCurrentDevice());
}

size_t getMaxSharedMemPerBlock(int device) {
  return getDeviceProperties(device).sharedMemPerBlock;
}

size_t getMaxSharedMemPerBlockCurrentDevice() {
  return getMaxSharedMemPerBlock(getCurrentDevice());
}

int getDeviceForAddress(const void* p) {
  if (!p) {
    return -1;
  }

  cudaPointerAttributes att;
  cudaError_t err = cudaPointerGetAttributes(&att, p);

  if (err == cudaErrorInvalidValue) {
    // Make sure the current thread error status has been reset
    err = cudaGetLastError();

    return -1;
  }

  // memoryType is deprecated for CUDA 10.0+
#if CUDA_VERSION < 10000
  if (att.memoryType == cudaMemoryTypeHost) {
    return -1;
  } else {
    return att.device;
  }
#else
  // FIXME: what to use for managed memory?
  if (att.type == cudaMemoryTypeDevice) {
    return att.device;
  } else {
    return -1;
  }
#endif
}

bool getFullUnifiedMemSupport(int device) {
  const auto& prop = getDeviceProperties(device);
  return (prop.major >= 6);
}

bool getFullUnifiedMemSupportCurrentDevice() {
  return getFullUnifiedMemSupport(getCurrentDevice());
}

DeviceScope::DeviceScope(int device) {
  if (device >= 0) {
    int curDevice = getCurrentDevice();

    if (curDevice != device) {
      prevDevice_ = curDevice;
      setCurrentDevice(device);
      return;
    }
  }

  // Otherwise, we keep the current device
  prevDevice_ = -1;
}

DeviceScope::~DeviceScope() {
  if (prevDevice_ != -1) {
    setCurrentDevice(prevDevice_);
  }
}

CudaEvent::CudaEvent(cudaStream_t stream, bool timer) : event_(nullptr) {
  CUDA_VERIFY(cudaEventCreateWithFlags(
      &event_, timer ? cudaEventDefault : cudaEventDisableTiming));
  CUDA_VERIFY(cudaEventRecord(event_, stream));
}

CudaEvent::CudaEvent(CudaEvent&& event) noexcept
    : event_(std::move(event.event_)) {
  event.event_ = nullptr;
}

CudaEvent::~CudaEvent() {
  if (event_) {
    CUDA_VERIFY(cudaEventDestroy(event_));
  }
}

CudaEvent& CudaEvent::operator=(CudaEvent&& event) noexcept {
  event_ = std::move(event.event_);
  event.event_ = nullptr;

  return *this;
}

void CudaEvent::streamWaitOnEvent(cudaStream_t stream) {
  CUDA_VERIFY(cudaStreamWaitEvent(stream, event_, 0));
}

void CudaEvent::cpuWaitOnEvent() {
  CUDA_VERIFY(cudaEventSynchronize(event_));
}

float CudaEvent::timeFrom(CudaEvent& from) {
  cpuWaitOnEvent();
  float ms = 0;
  CUDA_VERIFY(cudaEventElapsedTime(&ms, from.event_, event_));

  return ms;
}

CudaStream::CudaStream(int flags) : stream_(nullptr) {
  CUDA_VERIFY(cudaStreamCreateWithFlags(&stream_, flags));
}

CudaStream::CudaStream(CudaStream&& stream) noexcept
    : stream_(std::move(stream.stream_)) {
  stream.stream_ = nullptr;
}

CudaStream::~CudaStream() {
  if (stream_) {
    CUDA_VERIFY(cudaStreamDestroy(stream_));
  }
}

CudaStream& CudaStream::operator=(CudaStream&& stream) noexcept {
  stream_ = std::move(stream.stream_);
  stream.stream_ = nullptr;

  return *this;
}

CudaStream CudaStream::make() {
  return CudaStream();
}

CudaStream CudaStream::makeNonBlocking() {
  return CudaStream(cudaStreamNonBlocking);
}

} // namespace

#endif 
