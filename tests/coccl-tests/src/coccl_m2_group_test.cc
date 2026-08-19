#include "coccl_group.h"
#include "coccl_group_internal.h"
#include "coccl_prepared_call.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

std::vector<cocclOperation> executed;
std::vector<cocclOperation> replayed;

void fail(const char* expression, int line) {
  std::fprintf(stderr, "line %d: %s\n", line, expression);
  std::exit(1);
}

#define EXPECT(expression) \
  do { if (!(expression)) fail(#expression, __LINE__); } while (0)

cocclPreparedCall call(cocclOperation operation) {
  cocclPreparedCall prepared;
  prepared.info.operation = operation;
  prepared.compressors.handles[static_cast<size_t>(
      cocclCompressionScope::Default)] = reinterpret_cast<void*>(0x1);
  return prepared;
}

}  // namespace

ncclResult_t cocclExecutePreparedCall(const cocclPreparedCall* prepared) {
  executed.push_back(prepared->info.operation);
  return ncclSuccess;
}

ncclResult_t cocclReplayNativeCall(const cocclInfo& info) {
  replayed.push_back(info.operation);
  return ncclSuccess;
}

ncclResult_t cocclExecuteSendRecvBatch(
    const cocclPreparedCall*, size_t) {
  return ncclSuccess;
}

int main() {
  cocclPreparedCall first = call(cocclOperation::AllGather);
  cocclPreparedCall second = call(cocclOperation::AllToAll);
  EXPECT(cocclGroupEnqueue(&first) == ncclSuccess);
  EXPECT(cocclGroupEnqueue(&second) == ncclSuccess);
  first.info.operation = cocclOperation::AllReduce;
  EXPECT(cocclGroupPrepareEnd(false) == ncclSuccess);
  EXPECT(cocclGroupHasPending() && replayed.empty());
  EXPECT(cocclGroupDrain() == ncclSuccess);
  EXPECT(!cocclGroupHasPending() && executed.size() == 2);
  EXPECT(executed[0] == cocclOperation::AllGather);
  EXPECT(executed[1] == cocclOperation::AllToAll);

  executed.clear();
  EXPECT(cocclGroupEnqueue(&first) == ncclSuccess);
  EXPECT(cocclGroupEnqueue(&second) == ncclSuccess);
  EXPECT(cocclGroupPrepareEnd(true) == ncclSuccess);
  EXPECT(!cocclGroupHasPending() && replayed.size() == 2);
  EXPECT(replayed[0] == cocclOperation::AllReduce);
  EXPECT(replayed[1] == cocclOperation::AllToAll);
  EXPECT(cocclGroupDrain() == ncclSuccess && executed.empty());

  EXPECT(cocclGroupEnqueue(&first) == ncclSuccess);
  cocclGroupAbort();
  EXPECT(!cocclGroupHasPending());
  return 0;
}
