#include "core/config/coccl_config.h"
#include "core/config/coccl_config_debug.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

void fail(const char* expression, int line) {
  std::fprintf(stderr, "line %d: %s\n", line, expression);
  std::exit(1);
}

#define EXPECT(expression) \
  do { if (!(expression)) fail(#expression, __LINE__); } while (0)

struct ExpectedScope {
  bool enabled;
  const char* compressor;
  cocclCompressionScope source;
  const char* option;
  const char* value;
};

struct ConfigCase {
  const char* name;
  const char* body;
  std::array<ExpectedScope, 3> expected;
};

constexpr ExpectedScope disabled(cocclCompressionScope source) {
  return {false, "", source, "", ""};
}

constexpr ExpectedScope enabled(const char* compressor,
                                cocclCompressionScope source,
                                const char* option, const char* value) {
  return {true, compressor, source, option, value};
}

std::string configPreamble() {
  return
      "runtime.mode = \"normal\"\n"
      "runtime.compression_threshold_bytes = 4096\n"
      "compressor_plugins.compressors = [\"sdp4bit\", \"zfp\", \"dietgpu\"]\n"
      "compressor_plugins.library_path = \".\"\n"
      "normal.all_reduce.threshold_bytes = 123\n";
}

void writeFile(const std::string& path, const std::string& contents) {
  std::ofstream output(path);
  EXPECT(output.good());
  output << contents;
  EXPECT(output.good());
}

const cocclPrimitivePolicy& parse(const std::string& path,
                                  const std::string& contents,
                                  cocclConfig* config) {
  writeFile(path, contents);
  std::string error;
  EXPECT(cocclLoadConfigFile(path, config, &error));
  EXPECT(error.empty());
  return config->normal.allReduce;
}

void checkPolicy(const cocclPrimitivePolicy& policy,
                 const ConfigCase& test, bool checkSource = true) {
  EXPECT(policy.thresholdBytes == 123);
  for (size_t index = 0; index < test.expected.size(); ++index) {
    const cocclCompressionScope scope =
        static_cast<cocclCompressionScope>(index);
    const cocclEffectiveCompressorScope actual =
        cocclEffectiveCompressorScopeFor(policy, scope);
    const ExpectedScope& expected = test.expected[index];
    EXPECT(actual.enabled() == expected.enabled);
    if (checkSource) EXPECT(actual.source == expected.source);
    if (!expected.enabled) continue;
    EXPECT(actual.entry->name == expected.compressor);
    const auto option = actual.entry->values.find(expected.option);
    EXPECT(option != actual.entry->values.end());
    EXPECT(option->second == expected.value);
  }
}

void checkRoundTrip(const std::string& path, const cocclConfig& input,
                    const ConfigCase& test) {
  std::string formatted;
  for (const std::string& line : cocclFormatEffectiveConfig(input)) {
    formatted += line;
    formatted.push_back('\n');
  }
  cocclConfig reparsed;
  const cocclPrimitivePolicy& policy = parse(path, formatted, &reparsed);
  checkPolicy(policy, test, false);
}

}  // namespace

int main(int argc, char** argv) {
  EXPECT(argc >= 2);
  const std::array<ConfigCase, 10> cases = {{
      {"default-only",
       "normal.all_reduce.default.compressor = \"sdp4bit\"\n"
       "normal.all_reduce.default.config.quantBits = 4\n",
       {enabled("sdp4bit", cocclCompressionScope::Default, "quantBits", "4"),
        enabled("sdp4bit", cocclCompressionScope::Default, "quantBits", "4"),
        enabled("sdp4bit", cocclCompressionScope::Default, "quantBits", "4")}},
      {"intra-only",
       "normal.all_reduce.intra.compressor = \"sdp4bit\"\n"
       "normal.all_reduce.intra.config.quantBits = 8\n",
       {disabled(cocclCompressionScope::Default),
        enabled("sdp4bit", cocclCompressionScope::Intra, "quantBits", "8"),
        disabled(cocclCompressionScope::Default)}},
      {"inter-only",
       "normal.all_reduce.inter.compressor = \"zfp\"\n"
       "normal.all_reduce.inter.config.rate = 16\n",
       {disabled(cocclCompressionScope::Default),
        disabled(cocclCompressionScope::Default),
        enabled("zfp", cocclCompressionScope::Inter, "rate", "16")}},
      {"intra-inter",
       "normal.all_reduce.intra.compressor = \"sdp4bit\"\n"
       "normal.all_reduce.intra.config.quantBits = 8\n"
       "normal.all_reduce.inter.compressor = \"sdp4bit\"\n"
       "normal.all_reduce.inter.config.quantBits = 4\n",
       {disabled(cocclCompressionScope::Default),
        enabled("sdp4bit", cocclCompressionScope::Intra, "quantBits", "8"),
        enabled("sdp4bit", cocclCompressionScope::Inter, "quantBits", "4")}},
      {"default-intra-override",
       "normal.all_reduce.default.compressor = \"sdp4bit\"\n"
       "normal.all_reduce.default.config.quantBits = 4\n"
       "normal.all_reduce.intra.compressor = \"sdp4bit\"\n"
       "normal.all_reduce.intra.config.quantBits = 8\n",
       {enabled("sdp4bit", cocclCompressionScope::Default, "quantBits", "4"),
        enabled("sdp4bit", cocclCompressionScope::Intra, "quantBits", "8"),
        enabled("sdp4bit", cocclCompressionScope::Default, "quantBits", "4")}},
      {"default-inter-override",
       "normal.all_reduce.default.compressor = \"sdp4bit\"\n"
       "normal.all_reduce.default.config.quantBits = 8\n"
       "normal.all_reduce.inter.compressor = \"sdp4bit\"\n"
       "normal.all_reduce.inter.config.quantBits = 4\n",
       {enabled("sdp4bit", cocclCompressionScope::Default, "quantBits", "8"),
        enabled("sdp4bit", cocclCompressionScope::Default, "quantBits", "8"),
        enabled("sdp4bit", cocclCompressionScope::Inter, "quantBits", "4")}},
      {"disable-intra",
       "normal.all_reduce.default.compressor = \"sdp4bit\"\n"
       "normal.all_reduce.default.config.quantBits = 8\n"
       "normal.all_reduce.intra.enabled = false\n"
       "normal.all_reduce.inter.compressor = \"sdp4bit\"\n"
       "normal.all_reduce.inter.config.quantBits = 4\n",
       {enabled("sdp4bit", cocclCompressionScope::Default, "quantBits", "8"),
        disabled(cocclCompressionScope::Intra),
        enabled("sdp4bit", cocclCompressionScope::Inter, "quantBits", "4")}},
      {"disable-inter",
       "normal.all_reduce.default.compressor = \"sdp4bit\"\n"
       "normal.all_reduce.default.config.quantBits = 4\n"
       "normal.all_reduce.intra.compressor = \"sdp4bit\"\n"
       "normal.all_reduce.intra.config.quantBits = 8\n"
       "normal.all_reduce.inter.enabled = false\n",
       {enabled("sdp4bit", cocclCompressionScope::Default, "quantBits", "4"),
        enabled("sdp4bit", cocclCompressionScope::Intra, "quantBits", "8"),
        disabled(cocclCompressionScope::Inter)}},
      {"different-plugins",
       "normal.all_reduce.intra.compressor = \"sdp4bit\"\n"
       "normal.all_reduce.intra.config.quantBits = 8\n"
       "normal.all_reduce.inter.compressor = \"zfp\"\n"
       "normal.all_reduce.inter.config.rate = 16\n",
       {disabled(cocclCompressionScope::Default),
        enabled("sdp4bit", cocclCompressionScope::Intra, "quantBits", "8"),
        enabled("zfp", cocclCompressionScope::Inter, "rate", "16")}},
      {"all-disabled", "",
       {disabled(cocclCompressionScope::Default),
        disabled(cocclCompressionScope::Default),
        disabled(cocclCompressionScope::Default)}}}};

  for (const ConfigCase& test : cases) {
    const std::string base = std::string(argv[1]) + "/" + test.name;
    cocclConfig config;
    const cocclPrimitivePolicy& policy =
        parse(base + ".toml", configPreamble() + test.body, &config);
    checkPolicy(policy, test);
    checkRoundTrip(base + "-formatted.toml", config, test);
  }

  for (int index = 2; index < argc; ++index) {
    cocclConfig config;
    std::string error;
    EXPECT(cocclLoadConfigFile(argv[index], &config, &error));
    EXPECT(error.empty());
  }

  std::printf("coccl scope config: PASS\n");
  return 0;
}
