#ifndef COCCL_COMM_OP_H_
#define COCCL_COMM_OP_H_

// Operation tag used to select the compressor chain for a specific collective.
// Keep numeric values stable: runtime stores chains keyed by these tags, and
// env/config fallback rules rely on base op vs *_Inter/BWD relationships.
// *_Inter lists can use a different compressor/config for inter-node stages.
enum ncclCommOp {
  AlltoAll = 0,
  AlltoAll_Inter = 1,
  AllReduce = 2,
  AllReduce_Inter = 3,
  AllGather = 4,
  AllGather_Inter = 5,
  ReduceScatter = 6,
  ReduceScatter_Inter = 7,
  SendRecv = 8,
  SendRecv_BWD = 9
};

typedef ncclCommOp ncclCommOp_t;

#endif
