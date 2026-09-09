#include "runtime/coccl_operation.h"

#include <stddef.h>

namespace {

constexpr cocclOperationDescriptor kOperationDescriptors[] = {
    {cocclOperation::AllGather, "AllGather",
     cocclOperationTraitScaleBytesByRanks},
    {cocclOperation::ReduceScatter, "ReduceScatter",
     cocclOperationTraitScaleBytesByRanks | cocclOperationTraitReduction},
    // Every compressed AllReduce flow partitions the user buffer by rank.
    // Native NCCL remains the fallback for a partial final partition.
    {cocclOperation::AllReduce, "AllReduce",
     cocclOperationTraitReduction | cocclOperationTraitCountDivisibleByRanks},
    {cocclOperation::AllToAll, "AllToAll",
     cocclOperationTraitScaleBytesByRanks},
    {cocclOperation::SendRecv, "SendRecv", cocclOperationTraitNone},
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
