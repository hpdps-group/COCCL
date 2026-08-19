#ifndef COCCL_COMPRESSOR_PLUGIN_SDK_H_
#define COCCL_COMPRESSOR_PLUGIN_SDK_H_

#include "compressor_plugin/detail/coccl_compressor_abi.h"

#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace coccl {

using Status = ncclResult_t;

inline size_t dataTypeSize(ncclDataType_t datatype) {
  switch (datatype) {
    case ncclInt8:
    case ncclUint8:
      return 1;
    case ncclFloat16:
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
#endif
      return 2;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64:
      return 8;
    default:
      return 0;
  }
}

inline bool checkedAdd(size_t lhs, size_t rhs, size_t* result) {
  if (result == nullptr ||
      rhs > std::numeric_limits<size_t>::max() - lhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

inline bool checkedMultiply(size_t lhs, size_t rhs, size_t* result) {
  if (result == nullptr ||
      (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

inline Status fromCuda(cudaError_t result) {
  return result == cudaSuccess ? ncclSuccess : ncclUnhandledCudaError;
}

class Buffer {
 public:
  Buffer() = default;
  explicit Buffer(const cocclCompressorBufferView& view) : view_(view) {}

  void* data() const { return view_.data; }
  size_t bytes() const { return view_.bytes; }

  template <typename T>
  T* dataAs() const {
    return static_cast<T*>(view_.data);
  }

  explicit operator bool() const { return view_.data != nullptr; }

 private:
  cocclCompressorBufferView view_ = {};
};

class Input {
 public:
  explicit Input(const cocclCompressorView& view) : view_(&view) {}

  const void* data() const { return view_->data; }
  size_t bytes() const { return view_->bytes; }
  size_t elements() const { return view_->elements; }
  size_t chunks() const { return view_->chunks; }
  ncclDataType_t datatype() const { return view_->datatype; }
  bool framed() const { return view_->frameMetadata != nullptr; }
  const cocclCompressorFrameMetadata* frameMetadata() const {
    return view_->frameMetadata;
  }
  size_t frameStrideBytes() const { return view_->frameStrideBytes; }
  size_t elementsPerChunk() const {
    return view_->chunks == 0 ? 0 : view_->elements / view_->chunks;
  }

  template <typename T>
  const T* dataAs() const {
    return static_cast<const T*>(view_->data);
  }

 private:
  const cocclCompressorView* view_;
};

// Read-only raw shape that the requested encoding operation will produce.
class Shape {
 public:
  explicit Shape(const cocclCompressorSizeQuery* query) : query_(query) {}

  size_t bytes() const {
    size_t result = 0;
    return checkedMultiply(elements(), dataTypeSize(datatype()), &result)
        ? result : 0;
  }
  size_t elements() const { return query_->elements; }
  size_t chunks() const { return query_->chunks; }
  ncclDataType_t datatype() const { return query_->datatype; }
  size_t elementsPerChunk() const {
    return query_->chunks == 0 ? 0 : query_->elements / query_->chunks;
  }

 private:
  const cocclCompressorSizeQuery* query_;
};

class SizeContext {
 public:
  explicit SizeContext(const cocclCompressorSizeQuery* query)
      : query_(query) {}

  cocclCompressorOperation operation() const { return query_->operation; }

  template <typename Config>
  const Config& config() const {
    return *static_cast<const Config*>(query_->config);
  }

 private:
  const cocclCompressorSizeQuery* query_;
};

class Output {
 public:
  explicit Output(cocclCompressorView* view)
      : view_(view), plannedElements_(view == nullptr ? 0 : view->elements),
        plannedChunks_(view == nullptr ? 0 : view->chunks),
        plannedDatatype_(view == nullptr ? ncclInt8 : view->datatype) {}

  void* data() const { return view_ == nullptr ? nullptr : view_->data; }
  size_t capacityBytes() const {
    return view_ == nullptr ? 0 : view_->capacityBytes;
  }
  size_t elements() const { return plannedElements_; }
  size_t chunks() const { return plannedChunks_; }
  ncclDataType_t datatype() const { return plannedDatatype_; }
  bool framed() const {
    return view_ != nullptr && view_->frameMetadata != nullptr;
  }
  cocclCompressorFrameMetadata* frameMetadata() const {
    return view_ == nullptr ? nullptr : view_->frameMetadata;
  }
  size_t frameStrideBytes() const {
    return view_ == nullptr ? 0 : view_->frameStrideBytes;
  }

  template <typename T>
  T* dataAs() const {
    return static_cast<T*>(data());
  }

  Status commitPassthrough(const Input& input) {
    return commit(input.bytes(), COCCL_COMPRESSOR_RAW_PASSTHROUGH,
                  input.chunks());
  }

  Status passthrough(const Input& input, cudaStream_t stream) {
    if (view_ == nullptr || view_->data == nullptr ||
        input.data() == nullptr || input.bytes() > view_->capacityBytes) {
      return ncclInvalidArgument;
    }
    if (view_->data != input.data()) {
      const cudaError_t result = cudaMemcpyAsync(
          view_->data, input.data(), input.bytes(), cudaMemcpyDeviceToDevice,
          stream);
      if (result != cudaSuccess) return fromCuda(result);
    }
    return commitPassthrough(input);
  }

  Status commit(size_t elements, ncclDataType_t datatype, size_t chunks) {
    const size_t typeBytes = dataTypeSize(datatype);
    size_t bytes = 0;
    if (view_ == nullptr || view_->data == nullptr || chunks == 0 ||
        elements % chunks != 0 || typeBytes == 0 ||
        (elements != 0 && typeBytes > SIZE_MAX / elements)) {
      return ncclInvalidArgument;
    }
    bytes = elements * typeBytes;
    if (bytes > view_->capacityBytes) return ncclInvalidArgument;
    view_->bytes = bytes;
    view_->elements = elements;
    view_->chunks = chunks;
    view_->datatype = datatype;
    committed_ = true;
    return ncclSuccess;
  }

  Status commitBytes(size_t bytes, size_t chunks) {
    return commit(bytes, ncclInt8, chunks);
  }

  Status commitFrames() {
    size_t bytes = 0;
    if (view_ == nullptr || view_->frameMetadata == nullptr ||
        view_->frameStrideBytes == 0 || plannedChunks_ == 0 ||
        !checkedMultiply(view_->frameStrideBytes, plannedChunks_, &bytes) ||
        bytes > view_->capacityBytes) {
      return ncclInvalidArgument;
    }
    return commit(bytes, ncclInt8, plannedChunks_);
  }

  bool committed() const { return committed_; }

  Status commitPlanned() {
    return commit(plannedElements_, plannedDatatype_, plannedChunks_);
  }

 private:
  cocclCompressorView* view_ = nullptr;
  size_t plannedElements_ = 0;
  size_t plannedChunks_ = 0;
  ncclDataType_t plannedDatatype_ = ncclInt8;
  bool committed_ = false;
};

inline bool shouldPassthrough(const Input& input, size_t encodedBytes) {
  return encodedBytes > input.bytes();
}

inline bool isPassthrough(const Input& input) {
  return input.datatype() == COCCL_COMPRESSOR_RAW_PASSTHROUGH;
}

template <typename Enum>
struct EnumEntry {
  const char* name;
  Enum value;
};

class ConfigContext {
 public:
  explicit ConfigContext(const cocclCompressorConfigContext* context)
      : context_(context) {}

  bool hierarchical() const {
    return context_ != nullptr &&
        context_->variant == cocclCompressorConfigHierarchical;
  }
  int nodes() const { return context_ == nullptr ? 0 : context_->nodes; }
  int devicesPerNode() const {
    return context_ == nullptr ? 0 : context_->devicesPerNode;
  }

 private:
  const cocclCompressorConfigContext* context_;
};

class ConfigReader {
 public:
  ConfigReader(const cocclConfigView* view, char* error, size_t errorCapacity)
      : view_(view), error_(error), errorCapacity_(errorCapacity),
        consumed_(view == nullptr ? 0 : view->count, 0) {
    if (error_ != nullptr && errorCapacity_ != 0) error_[0] = '\0';
  }

  template <typename Value>
  ConfigReader& get(const char* key, Value& output) {
    if (!ready(key)) return *this;
    bool found = false;
    const char* value = consume(key, &found);
    if (status_ == ncclSuccess && found && !parse(value, &output)) {
      fail(key, "has an invalid value");
    }
    return *this;
  }

  template <typename Value>
  ConfigReader& get(const char* key, Value& output, Value minimum,
                    Value maximum) {
    if (!ready(key) || maximum < minimum) {
      if (status_ == ncclSuccess) fail(key, "has an invalid range");
      return *this;
    }
    bool found = false;
    const char* value = consume(key, &found);
    if (status_ == ncclSuccess && found &&
        (!parse(value, &output) || output < minimum || output > maximum)) {
      fail(key, "is outside the allowed range");
    }
    return *this;
  }

  template <typename Enum>
  ConfigReader& getEnum(
      const char* key, Enum& output,
      std::initializer_list<EnumEntry<Enum>> entries) {
    if (!ready(key)) return *this;
    bool found = false;
    const char* value = consume(key, &found);
    if (status_ != ncclSuccess || !found) return *this;
    for (const auto& entry : entries) {
      if (entry.name != nullptr && strcmp(value, entry.name) == 0) {
        output = entry.value;
        return *this;
      }
    }
    fail(key, "is not a supported enum value");
    return *this;
  }

  Status finish() {
    if (status_ != ncclSuccess) return status_;
    if (view_ == nullptr) return ncclSuccess;
    for (size_t i = 0; i < consumed_.size(); ++i) {
      if (consumed_[i] == 0) {
        const char* key = view_->pairs == nullptr ? nullptr
                                                  : view_->pairs[i].key;
        fail(key, "is not recognized by this compressor");
        break;
      }
    }
    return status_;
  }

 private:
  bool ready(const char* key) {
    if (status_ != ncclSuccess) return false;
    if (key == nullptr) {
      fail(nullptr, "has a null key");
      return false;
    }
    return true;
  }

  void fail(const char* key, const char* reason) {
    if (status_ != ncclSuccess) return;
    status_ = ncclInvalidArgument;
    if (error_ != nullptr && errorCapacity_ != 0) {
      snprintf(error_, errorCapacity_, "%s%s%s",
               key == nullptr ? "configuration" : key,
               reason == nullptr ? "" : " ", reason == nullptr ? "" : reason);
    }
  }

  const char* consume(const char* key, bool* found) {
    *found = false;
    if (view_ == nullptr) return nullptr;
    if (view_->count != 0 && view_->pairs == nullptr) {
      fail(nullptr, "has no key/value storage");
      return nullptr;
    }
    for (size_t i = 0; i < view_->count; ++i) {
      const cocclConfigPair& pair = view_->pairs[i];
      if (pair.key != nullptr && strcmp(pair.key, key) == 0) {
        if (consumed_[i] != 0 || pair.value == nullptr) {
          fail(key, "is duplicated or has no value");
          return nullptr;
        }
        consumed_[i] = 1;
        *found = true;
        return pair.value;
      }
    }
    return nullptr;
  }

  template <typename Value,
            typename std::enable_if<std::is_integral<Value>::value &&
                                        !std::is_same<Value, bool>::value,
                                    int>::type = 0>
  static bool parse(const char* text, Value* output) {
    if (text == nullptr || text[0] == '\0') return false;
    Value parsed{};
    const char* end = text + strlen(text);
    auto result = std::from_chars(text, end, parsed, 10);
    if (result.ec != std::errc() || result.ptr != end) return false;
    *output = parsed;
    return true;
  }

  template <typename Value,
            typename std::enable_if<std::is_floating_point<Value>::value,
                                    int>::type = 0>
  static bool parse(const char* text, Value* output) {
    if (text == nullptr || text[0] == '\0') return false;
    errno = 0;
    char* end = nullptr;
    double parsed = strtod(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' ||
        !std::isfinite(parsed) ||
        parsed < -std::numeric_limits<Value>::max() ||
        parsed > std::numeric_limits<Value>::max()) {
      return false;
    }
    *output = static_cast<Value>(parsed);
    return true;
  }

  static bool parse(const char* text, bool* output) {
    if (text != nullptr && strcmp(text, "true") == 0) {
      *output = true;
      return true;
    }
    if (text != nullptr && strcmp(text, "false") == 0) {
      *output = false;
      return true;
    }
    return false;
  }

  static bool parse(const char* text, std::string* output) {
    if (text == nullptr) return false;
    *output = text;
    return true;
  }

  const cocclConfigView* view_ = nullptr;
  char* error_ = nullptr;
  size_t errorCapacity_ = 0;
  std::vector<unsigned char> consumed_;
  Status status_ = ncclSuccess;
};

class Context {
 public:
  Context(cocclCompressorCall* call) : call_(call) {}

  cudaStream_t stream() const { return call_->execution->stream; }
  int cudaDevice() const { return call_->execution->cudaDev; }
  int rank() const { return call_->execution->rank; }
  int ranks() const { return call_->execution->nRanks; }
  int nodes() const { return call_->execution->nodes; }
  int devicesPerNode() const { return call_->execution->devicesPerNode; }
  size_t reduceChunks() const { return call_->reduceChunks; }
  ncclDataType_t originalDatatype() const {
    return call_->originalDatatype;
  }
  size_t originalElements() const { return call_->originalElements; }

  template <typename Config>
  const Config& config() const {
    return *static_cast<const Config*>(call_->config);
  }

  Status scratch(size_t bytes, Buffer* buffer) {
    cocclCompressorBufferView view = {};
    Status result = call_->execution->hostApi->allocateScratch(
        call_->execution->hostContext, bytes, &view);
    if (result == ncclSuccess) *buffer = Buffer(view);
    return result;
  }

  Status persistent(size_t slot, size_t bytes, Buffer* buffer) {
    cocclCompressorBufferView view = {};
    Status result = call_->execution->hostApi->acquirePersistent(
        call_->execution->hostContext, slot, bytes, &view);
    if (result == ncclSuccess) *buffer = Buffer(view);
    return result;
  }

  template <typename State>
  Status instance(State** state) {
    static_assert(std::is_default_constructible<State>::value,
                  "compressor State must be default constructible");
    static_assert(std::is_destructible<State>::value,
                  "compressor State must be destructible");
    static const unsigned char typeKey = 0;
    void* opaque = nullptr;
    Status result = call_->execution->hostApi->getOrCreateState(
        call_->execution->hostContext, &typeKey, &createState<State>,
        &destroyState<State>, &opaque);
    if (result == ncclSuccess) *state = static_cast<State*>(opaque);
    return result;
  }

 private:
  template <typename State>
  static Status createState(void** state) {
    *state = new (std::nothrow) State();
    return *state == nullptr ? ncclSystemError : ncclSuccess;
  }

  template <typename State>
  static void destroyState(void* state) {
    delete static_cast<State*>(state);
  }

  cocclCompressorCall* call_;
};

namespace detail {

struct EmptyConfig {};

template <typename...>
using VoidT = void;

template <typename Compressor, typename = void>
struct ConfigTraits {
  using Type = EmptyConfig;
  static constexpr bool present = false;
};

template <typename Compressor>
struct ConfigTraits<Compressor, VoidT<typename Compressor::Config>> {
  using Type = typename Compressor::Config;
  static constexpr bool present = true;
};

template <typename Compressor, typename = void>
struct HasConfigure : std::false_type {};

template <typename Compressor>
struct HasConfigure<
    Compressor,
    VoidT<decltype(Compressor::configure(
        std::declval<ConfigReader&>(),
        std::declval<typename ConfigTraits<Compressor>::Type&>(),
        std::declval<const ConfigContext&>()))>> : std::true_type {};

template <typename Compressor, typename = void>
struct HasCompress : std::false_type {};

template <typename Compressor>
struct HasCompress<
    Compressor,
    VoidT<decltype(Compressor::compress(
        std::declval<const Input&>(), std::declval<Output&>(),
        std::declval<Context&>()))>> : std::true_type {};

template <typename Compressor, typename = void>
struct HasDecompress : std::false_type {};

template <typename Compressor>
struct HasDecompress<
    Compressor,
    VoidT<decltype(Compressor::decompress(
        std::declval<const Input&>(), std::declval<Output&>(),
        std::declval<Context&>()))>> : std::true_type {};

template <typename Compressor, typename = void>
struct HasDecompressReduce : std::false_type {};

template <typename Compressor>
struct HasDecompressReduce<
    Compressor,
    VoidT<decltype(Compressor::decompressReduce(
        std::declval<const Input&>(), std::declval<Output&>(),
        std::declval<Context&>()))>> : std::true_type {};

template <typename Compressor, typename = void>
struct HasDecompressReduceCompress : std::false_type {};

template <typename Compressor>
struct HasDecompressReduceCompress<
    Compressor,
    VoidT<decltype(Compressor::decompressReduceCompress(
        std::declval<const Input&>(), std::declval<Output&>(),
        std::declval<Context&>()))>> : std::true_type {};

template <typename Compressor, typename = void>
struct HasEncodedSizeBound : std::false_type {};

template <typename Compressor>
struct HasEncodedSizeBound<
    Compressor,
    VoidT<decltype(Compressor::encodedSizeBound(
        std::declval<const Shape&>(), std::declval<size_t*>(),
        std::declval<const SizeContext&>()))>> : std::true_type {};

template <typename Compressor, typename = void>
struct FramedTraits {
  static constexpr bool value = false;
};

template <typename Compressor>
struct FramedTraits<Compressor, VoidT<decltype(Compressor::kFramed)>> {
  static constexpr bool value = Compressor::kFramed;
};

template <typename Compressor, typename = void>
struct FusedHierarchicalSwizzleTraits {
  static constexpr bool value = false;
};

template <typename Compressor>
struct FusedHierarchicalSwizzleTraits<
    Compressor,
    VoidT<decltype(Compressor::kFusedHierarchicalSwizzle)>> {
  static constexpr bool value = Compressor::kFusedHierarchicalSwizzle;
};

template <typename Compressor>
struct PluginAdapter {
  using Config = typename ConfigTraits<Compressor>::Type;

  static_assert(HasCompress<Compressor>::value,
                "a COCCL compressor must implement compress");
  static_assert(HasDecompress<Compressor>::value,
                "a COCCL compressor must implement decompress");
  static_assert(!HasConfigure<Compressor>::value ||
                    ConfigTraits<Compressor>::present,
                "configure requires a nested Compressor::Config type");

  static constexpr uint64_t capabilities() {
    return COCCL_COMPRESSOR_REQUIRED_CAPABILITIES |
        (HasDecompressReduce<Compressor>::value
             ? cocclCompressorCapabilityDecompressReduce : 0) |
        (HasDecompressReduceCompress<Compressor>::value
             ? cocclCompressorCapabilityDecompressReduceCompress : 0) |
        (FramedTraits<Compressor>::value
             ? cocclCompressorCapabilityFramed : 0) |
        (FusedHierarchicalSwizzleTraits<Compressor>::value
             ? cocclCompressorCapabilityFusedHierarchicalSwizzle : 0);
  }

  static Status execute(cocclCompressorCall* call) {
    if (call == nullptr || call->structSize != sizeof(*call) ||
        call->execution == nullptr ||
        call->execution->structSize != sizeof(*call->execution)) {
      return ncclInvalidArgument;
    }
    const cocclCompressorHostApi* hostApi = call->execution->hostApi;
    if (hostApi == nullptr ||
        hostApi->abiVersion != COCCL_COMPRESSOR_HOST_API_VERSION ||
        hostApi->structSize != sizeof(*hostApi) ||
        hostApi->allocateScratch == nullptr ||
        hostApi->acquirePersistent == nullptr ||
        hostApi->getOrCreateState == nullptr) {
      return ncclInvalidArgument;
    }
    const bool encodedOutput =
        call->operation == cocclCompressorOperationCompress ||
        call->operation ==
            cocclCompressorOperationDecompressReduceCompress;
    Input input(call->input);
    Output output(call->output);
    Context context(call);
    Status result = ncclInvalidUsage;
    switch (call->operation) {
      case cocclCompressorOperationCompress:
        result = Compressor::compress(input, output, context);
        break;
      case cocclCompressorOperationDecompress:
        result = Compressor::decompress(input, output, context);
        break;
      case cocclCompressorOperationDecompressReduce:
        result = executeDecompressReduce(input, output, context);
        break;
      case cocclCompressorOperationDecompressReduceCompress:
        result = executeDecompressReduceCompress(input, output, context);
        break;
      default:
        return ncclInvalidArgument;
    }
    if (result != ncclSuccess) return result;
    if (encodedOutput) return output.committed() ? ncclSuccess
                                                  : ncclInvalidUsage;
    return output.committed() ? ncclSuccess : output.commitPlanned();
  }

  static Status parseConfig(const cocclConfigView* values,
                            const cocclCompressorConfigContext* context,
                            void** parsedConfig, char* error,
                            size_t errorCapacity) {
    if (parsedConfig == nullptr || context == nullptr) {
      return ncclInvalidArgument;
    }
    *parsedConfig = nullptr;
    ConfigReader reader(values, error, errorCapacity);
    if constexpr (!ConfigTraits<Compressor>::present) {
      return reader.finish();
    } else {
      Config* config = new (std::nothrow) Config();
      if (config == nullptr) return ncclSystemError;
      Status result = ncclSuccess;
      if constexpr (HasConfigure<Compressor>::value) {
        const ConfigContext publicContext(context);
        result = Compressor::configure(reader, *config, publicContext);
      } else {
        result = reader.finish();
      }
      if (result == ncclSuccess) result = reader.finish();
      if (result != ncclSuccess) {
        delete config;
        return result;
      }
      *parsedConfig = config;
      return ncclSuccess;
    }
  }

  static void destroyConfig(void* config) {
    if constexpr (ConfigTraits<Compressor>::present) {
      delete static_cast<Config*>(config);
    }
  }

  static Status getEncodedSizeBound(
      const cocclCompressorSizeQuery* query, size_t* encodedBytes) {
    const Shape shape(query);
    const SizeContext context(query);
    return executeEncodedSizeBound(shape, encodedBytes, context);
  }

  static const cocclCompressorPlugin* descriptor(const char* name) {
    static const cocclCompressorPlugin plugin = {
        COCCL_COMPRESSOR_ABI_VERSION,
        sizeof(cocclCompressorPlugin),
        name,
        capabilities(),
        &execute,
        &parseConfig,
        &destroyConfig,
        &getEncodedSizeBound,
    };
    return &plugin;
  }

 private:
  template <typename T = Compressor>
  static typename std::enable_if<HasEncodedSizeBound<T>::value,
                                 Status>::type
  executeEncodedSizeBound(const Shape& shape, size_t* encodedBytes,
                          const SizeContext& context) {
    return T::encodedSizeBound(shape, encodedBytes, context);
  }

  template <typename T = Compressor>
  static typename std::enable_if<!HasEncodedSizeBound<T>::value,
                                 Status>::type
  executeEncodedSizeBound(const Shape&, size_t*, const SizeContext&) {
    return ncclInvalidUsage;
  }

  template <typename T = Compressor>
  static typename std::enable_if<HasDecompressReduce<T>::value,
                                 Status>::type
  executeDecompressReduce(const Input& input, Output& output,
                          Context& context) {
    return T::decompressReduce(input, output, context);
  }

  template <typename T = Compressor>
  static typename std::enable_if<!HasDecompressReduce<T>::value,
                                 Status>::type
  executeDecompressReduce(const Input&, Output&, Context&) {
    return ncclInvalidUsage;
  }

  template <typename T = Compressor>
  static typename std::enable_if<
      HasDecompressReduceCompress<T>::value, Status>::type
  executeDecompressReduceCompress(const Input& input, Output& output,
                                  Context& context) {
    return T::decompressReduceCompress(input, output, context);
  }

  template <typename T = Compressor>
  static typename std::enable_if<
      !HasDecompressReduceCompress<T>::value, Status>::type
  executeDecompressReduceCompress(const Input&, Output&, Context&) {
    return ncclInvalidUsage;
  }
};

}  // namespace detail
}  // namespace coccl

#if defined(__GNUC__)
#define COCCL_PLUGIN_EXPORT __attribute__((visibility("default")))
#else
#define COCCL_PLUGIN_EXPORT
#endif

#define COCCL_REGISTER_COMPRESSOR(pluginName, CompressorType)                \
  extern "C" COCCL_PLUGIN_EXPORT const cocclCompressorPlugin*               \
  cocclGetCompressorPlugin() {                                               \
    return ::coccl::detail::PluginAdapter<CompressorType>::descriptor(       \
        pluginName);                                                         \
  }

#endif
