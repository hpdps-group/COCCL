#include "runtime/coccl_operation.h"

#include <stddef.h>

namespace {

constexpr cocclOperationDescriptor kOperationDescriptors[] = {
    {cocclOperation::AllGather, "AllGather",
     cocclOperationTraitScaleBytesByRanks | cocclOperationTraitGrouped},
    {cocclOperation::ReduceScatter, "ReduceScatter",
     cocclOperationTraitScaleBytesByRanks | cocclOperationTraitReduction |
         cocclOperationTraitGrouped |
         cocclOperationTraitHierarchicalPolicy},
    // Every compressed AllReduce flow partitions the user buffer by rank.
    // Native NCCL remains the fallback for a partial final partition.
    {cocclOperation::AllReduce, "AllReduce",
     cocclOperationTraitReduction | cocclOperationTraitGrouped |
         cocclOperationTraitCountDivisibleByRanks |
         cocclOperationTraitHierarchicalPolicy},
    {cocclOperation::AllToAll, "AllToAll",
     cocclOperationTraitScaleBytesByRanks | cocclOperationTraitGrouped},
    {cocclOperation::SendRecv, "SendRecv",
     cocclOperationTraitDirectionalPolicy},
};

static_assert(sizeof(kOperationDescriptors) /
                      sizeof(kOperationDescriptors[0]) ==
                  static_cast<size_t>(cocclOperation::Count),
              "every COCCL operation requires one descriptor");

}  // namespace

const cocclOperationDescriptor* cocclGetOperationDescriptor(
    cocclOperation operation) {
  const size_t index = static_cast<size_t>(operation);
  return index < static_cast<size_t>(cocclOperation::Count)
      ? &kOperationDescriptors[index]
      : nullptr;
}

bool cocclOperationSupportsPolicy(const cocclOperationDescriptor* descriptor,
                                  cocclPolicyVariant variant) {
  if (descriptor == nullptr) return false;
  switch (variant) {
    case cocclPolicyVariant::Default:
      return true;
    case cocclPolicyVariant::Hierarchical:
      return cocclOperationHasTrait(
          descriptor, cocclOperationTraitHierarchicalPolicy);
    case cocclPolicyVariant::Forward:
    case cocclPolicyVariant::Backward:
      return cocclOperationHasTrait(
          descriptor, cocclOperationTraitDirectionalPolicy);
    default:
      return false;
  }
}
