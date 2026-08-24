#ifndef COCCL_OPERATION_H_
#define COCCL_OPERATION_H_

#include <stdint.h>

enum class cocclOperation : uint8_t {
  AllGather = 0,
  ReduceScatter,
  AllReduce,
  AllToAll,
  SendRecv,
  Count,
};

enum class cocclPolicyVariant : uint8_t {
  Default = 0,
  Forward,
  Backward,
};

enum class cocclCompressionScope : uint8_t {
  Default = 0,
  Intra,
  Inter,
  Count,
};

enum cocclOperationTrait : uint32_t {
  cocclOperationTraitNone = 0,
  cocclOperationTraitScaleBytesByRanks = 1u << 0,
  cocclOperationTraitReduction = 1u << 1,
  cocclOperationTraitGrouped = 1u << 2,
  cocclOperationTraitCountDivisibleByRanks = 1u << 3,
  cocclOperationTraitHierarchicalPolicy = 1u << 4,
  cocclOperationTraitDirectionalPolicy = 1u << 5,
};

struct cocclPolicyKey {
  cocclOperation operation = cocclOperation::Count;
  cocclPolicyVariant variant = cocclPolicyVariant::Default;
  cocclCompressionScope scope = cocclCompressionScope::Default;
};

inline bool operator==(const cocclPolicyKey& lhs,
                       const cocclPolicyKey& rhs) {
  return lhs.operation == rhs.operation && lhs.variant == rhs.variant &&
      lhs.scope == rhs.scope;
}

inline bool operator<(const cocclPolicyKey& lhs,
                      const cocclPolicyKey& rhs) {
  if (lhs.operation != rhs.operation) {
    return static_cast<uint8_t>(lhs.operation) <
           static_cast<uint8_t>(rhs.operation);
  }
  if (lhs.variant != rhs.variant) {
    return static_cast<uint8_t>(lhs.variant) <
           static_cast<uint8_t>(rhs.variant);
  }
  return static_cast<uint8_t>(lhs.scope) <
         static_cast<uint8_t>(rhs.scope);
}

constexpr cocclPolicyKey cocclDefaultPolicy(
    cocclOperation operation,
    cocclCompressionScope scope = cocclCompressionScope::Default) {
  return {operation, cocclPolicyVariant::Default, scope};
}

constexpr cocclPolicyKey cocclDirectionalPolicy(cocclOperation operation,
                                                 bool forward,
                                                 cocclCompressionScope scope =
                                                     cocclCompressionScope::Default) {
  return {operation, forward ? cocclPolicyVariant::Forward
                             : cocclPolicyVariant::Backward,
          scope};
}

constexpr cocclPolicyKey cocclPolicyForScope(
    cocclPolicyKey key, cocclCompressionScope scope) {
  key.scope = scope;
  return key;
}

// One descriptor is the authoritative runtime contract for each operation.
// Configuration bindings carry only a policy key; routing, shape checks,
// and grouping use these descriptors. Execution dispatch remains in the
// runtime translation unit so metadata-only users do not link CUDA primitives.
struct cocclOperationDescriptor {
  cocclOperation operation;
  const char* name;
  uint32_t traits;
};

const cocclOperationDescriptor* cocclGetOperationDescriptor(
    cocclOperation operation);
inline bool cocclOperationHasTrait(
    const cocclOperationDescriptor* descriptor, cocclOperationTrait trait) {
  return descriptor != nullptr && (descriptor->traits & (uint32_t)trait) != 0;
}
bool cocclOperationSupportsPolicy(const cocclOperationDescriptor* descriptor,
                                  cocclPolicyVariant variant);

#endif
