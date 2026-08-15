#include "compressor_plugin/coccl_compressor_plugin.h"

#include <stdio.h>
#include <string.h>

#include <string>

namespace {

enum class TestMode {
  Fast,
  Accurate,
};

int testTypedValues() {
  const cocclConfigPair pairs[] = {
      {"count", "128"},
      {"enabled", "true"},
      {"ratio", "0.25"},
      {"name", "sdp4bit"},
      {"mode", "Accurate"},
  };
  const cocclConfigView view = {pairs, sizeof(pairs) / sizeof(pairs[0])};
  int count = 1;
  bool enabled = false;
  double ratio = 1.0;
  std::string name;
  TestMode mode = TestMode::Fast;
  char error[128] = {};
  ncclResult_t result =
      coccl::ConfigReader(&view, error, sizeof(error))
          .get("count", count, 1, 256)
          .get("enabled", enabled)
          .get("ratio", ratio, 0.0, 1.0)
          .get("name", name)
          .getEnum("mode", mode,
                   {{"Fast", TestMode::Fast},
                    {"Accurate", TestMode::Accurate}})
          .finish();
  if (result != ncclSuccess || count != 128 || !enabled || ratio != 0.25 ||
      name != "sdp4bit" || mode != TestMode::Accurate) {
    fprintf(stderr, "typed config parsing failed\n");
    return 1;
  }
  return 0;
}

int testMissingValueRetainsDefault() {
  const cocclConfigView empty = {nullptr, 0};
  int count = 7;
  char error[128] = {};
  if (coccl::ConfigReader(&empty, error, sizeof(error))
          .get("count", count, 1, 16).finish() !=
          ncclSuccess ||
      count != 7) {
    fprintf(stderr, "missing config value changed its default\n");
    return 1;
  }
  return 0;
}

int testRejectedValues() {
  const cocclConfigPair outOfRange[] = {{"count", "17"}};
  const cocclConfigView outOfRangeView = {outOfRange, 1};
  int count = 0;
  char error[128] = {};
  if (coccl::ConfigReader(&outOfRangeView, error, sizeof(error))
          .get("count", count, 1, 16).finish() != ncclInvalidArgument ||
      strstr(error, "count") == nullptr) {
    fprintf(stderr, "out-of-range config value was accepted\n");
    return 1;
  }

  const cocclConfigPair unknown[] = {{"typo", "1"}};
  const cocclConfigView unknownView = {unknown, 1};
  if (coccl::ConfigReader(&unknownView, error, sizeof(error)).finish() !=
          ncclInvalidArgument ||
      strstr(error, "typo") == nullptr) {
    fprintf(stderr, "unknown config key was accepted\n");
    return 1;
  }

  const cocclConfigPair duplicate[] = {{"count", "1"}, {"count", "2"}};
  const cocclConfigView duplicateView = {duplicate, 2};
  if (coccl::ConfigReader(&duplicateView, error, sizeof(error))
          .get("count", count).finish() != ncclInvalidArgument) {
    fprintf(stderr, "duplicate config key was accepted\n");
    return 1;
  }

  const cocclConfigPair badEnum[] = {{"mode", "Unknown"}};
  const cocclConfigView badEnumView = {badEnum, 1};
  TestMode mode = TestMode::Fast;
  if (coccl::ConfigReader(&badEnumView, error, sizeof(error))
          .getEnum("mode", mode,
                   {{"Fast", TestMode::Fast},
                    {"Accurate", TestMode::Accurate}})
          .finish() != ncclInvalidArgument) {
    fprintf(stderr, "invalid enum config value was accepted\n");
    return 1;
  }
  return 0;
}

}  // namespace

int main() {
  if (testTypedValues() || testMissingValueRetainsDefault() ||
      testRejectedValues()) {
    return 1;
  }
  printf("COCCL config reader tests passed\n");
  return 0;
}
