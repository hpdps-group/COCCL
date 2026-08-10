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
  Hierarchical,
  Forward,
  Backward,
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
};

inline bool operator==(const cocclPolicyKey& lhs,
                       const cocclPolicyKey& rhs) {
  return lhs.operation == rhs.operation && lhs.variant == rhs.variant;
}

inline bool operator<(const cocclPolicyKey& lhs,
                      const cocclPolicyKey& rhs) {
  if (lhs.operation != rhs.operation) {
    return static_cast<uint8_t>(lhs.operation) <
           static_cast<uint8_t>(rhs.operation);
  }
  return static_cast<uint8_t>(lhs.variant) <
         static_cast<uint8_t>(rhs.variant);
}

constexpr cocclPolicyKey cocclDefaultPolicy(cocclOperation operation) {
  return {operation, cocclPolicyVariant::Default};
}

constexpr cocclPolicyKey cocclHierarchicalPolicy(cocclOperation operation) {
  return {operation, cocclPolicyVariant::Hierarchical};
}

constexpr cocclPolicyKey cocclDirectionalPolicy(cocclOperation operation,
                                                 bool forward) {
  return {operation, forward ? cocclPolicyVariant::Forward
                             : cocclPolicyVariant::Backward};
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
