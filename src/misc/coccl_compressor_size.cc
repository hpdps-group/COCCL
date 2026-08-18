#include "compress.h"

#include "collectives.h"

ncclResult_t cocclQueryCompressorEncodedSizeBound(
    const cocclCompressorPlugin* plugin, const void* config,
    cocclCompressorOperation operation, size_t elements, size_t chunks,
    ncclDataType_t datatype, size_t* encodedBytes) {
  if (encodedBytes == nullptr || plugin == nullptr || elements == 0 ||
      chunks == 0 || elements % chunks != 0 ||
      (operation != cocclCompressorOperationCompress &&
       operation != cocclCompressorOperationDecompressReduceCompress)) {
    return ncclInvalidArgument;
  }

  const int typeBytes = ncclTypeSize(datatype);
  if (typeBytes <= 0 || elements > SIZE_MAX / (size_t)typeBytes) {
    return ncclInvalidArgument;
  }
  const size_t rawBytes = elements * (size_t)typeBytes;
  *encodedBytes = rawBytes;
  if (plugin->getEncodedSizeBound == nullptr) return ncclSuccess;

  const cocclCompressorSizeQuery query = {
      sizeof(cocclCompressorSizeQuery), operation,
      elements, chunks, datatype, config};
  size_t pluginBytes = 0;
  const ncclResult_t result =
      plugin->getEncodedSizeBound(&query, &pluginBytes);
  if (result == ncclInvalidUsage) return ncclSuccess;
  if (result != ncclSuccess) return result;
  if (pluginBytes == 0) return ncclInvalidArgument;
  if (pluginBytes % chunks != 0) return ncclInvalidUsage;
  *encodedBytes = pluginBytes;
  return ncclSuccess;
}
