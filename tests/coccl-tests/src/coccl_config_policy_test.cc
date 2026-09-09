#include "core/config/coccl_config.h"
#include "core/compression/coccl_compressor_internal.h"
#include "debug.h"

#include <dlfcn.h>

#include <stdio.h>

#include <string>
#include <vector>
#include <utility>

namespace {

std::vector<std::pair<void*, cocclPolicyKey>> registrations;

bool valueIs(const cocclConfigValues& values, const char* key,
             const char* expected) {
  auto value = values.find(key);
  return value != values.end() && value->second == expected;
}

const cocclCompressorScopeEntry& scope(
    const cocclPrimitivePolicy& policy, cocclCompressionScope value) {
  return cocclConfiguredCompressorScope(policy, value);
}

int checkCatalog(const cocclConfig& config, const char* compressor) {
  return config.plugins.compressors.size() == 1 &&
         config.plugins.compressors[0] == compressor &&
         config.plugins.libraryPath.find(
             "build/obj/coccl-extend/compressor_plugin/libcompress") !=
             std::string::npos
      ? 0 : 1;
}

int checkSdp(const char* path) {
  cocclConfig config;
  std::string error;
  if (!cocclLoadConfigFile(path, &config, &error)) {
    fprintf(stderr, "failed to parse SDP4Bit TOML: %s\n", error.c_str());
    return 1;
  }
  if (checkCatalog(config, "sdp4bit") ||
      config.runtime.compressionThresholdBytes != 1048576 ||
      scope(config.normal.allToAll, cocclCompressionScope::Default).name !=
          "sdp4bit" ||
      scope(config.normal.allGather, cocclCompressionScope::Default).name !=
          "sdp4bit" ||
      scope(config.normal.allReduce, cocclCompressionScope::Default).name !=
          "sdp4bit" ||
      scope(config.normal.reduceScatter,
            cocclCompressionScope::Default).name != "sdp4bit") {
    fprintf(stderr, "SDP4Bit policy mapping is incomplete\n");
    return 1;
  }
  const auto& a2a = scope(
      config.normal.allToAll, cocclCompressionScope::Default).values;
  const auto& ar = scope(
      config.normal.allReduce, cocclCompressionScope::Default).values;
  const auto& arInter = scope(
      config.normal.allReduce, cocclCompressionScope::Inter).values;
  const auto& rs = scope(
      config.normal.reduceScatter, cocclCompressionScope::Default).values;
  if (!valueIs(a2a, "groupCount", "2048") ||
      !valueIs(a2a, "quantBits", "4") ||
      !valueIs(a2a, "quantType", "Symmetric") ||
      !valueIs(ar, "groupCount", "128") ||
      !valueIs(ar, "hadamard", "false") ||
      !valueIs(arInter, "quantBits", "4") ||
      !valueIs(arInter, "groupCount", "128") ||
      !valueIs(arInter, "hadamard", "false") ||
      arInter.count("inQuantBits") != 0 ||
      arInter.count("inGroupCount") != 0 ||
      !valueIs(rs, "groupCount", "128") ||
      !valueIs(rs, "hadamard", "true")) {
    fprintf(stderr, "SDP4Bit public TOML values are invalid\n");
    return 1;
  }
  return 0;
}

int checkZfp(const char* path) {
  cocclConfig config;
  std::string error;
  if (!cocclLoadConfigFile(path, &config, &error)) {
    fprintf(stderr, "failed to parse ZFP TOML: %s\n", error.c_str());
    return 1;
  }
  if (checkCatalog(config, "zfp") ||
      scope(config.normal.allToAll, cocclCompressionScope::Default).name !=
          "zfp" ||
      scope(config.normal.allGather, cocclCompressionScope::Default).name !=
          "zfp" ||
      scope(config.normal.allReduce, cocclCompressionScope::Default).name !=
          "zfp" ||
      scope(config.normal.reduceScatter,
            cocclCompressionScope::Default).name != "zfp" ||
      !valueIs(scope(config.normal.allToAll,
                     cocclCompressionScope::Default).values,
               "rate", "4") ||
      !valueIs(scope(config.normal.allGather,
                     cocclCompressionScope::Default).values,
               "rate", "8") ||
      !valueIs(scope(config.normal.allReduce,
                     cocclCompressionScope::Default).values,
               "rate", "8") ||
      !valueIs(scope(config.normal.reduceScatter,
                     cocclCompressionScope::Default).values,
               "rate", "8")) {
    fprintf(stderr, "ZFP public TOML values are invalid\n");
    return 1;
  }
  return 0;
}

void closeRegistry(cocclCompressorInternal::Registry* registry) {
  for (const auto& policy : registry->ownedPolicies) {
    policy->plugin->destroyConfig(policy->config);
  }
  registry->ownedPolicies.clear();
  for (const auto& plugin : registry->loadedPlugins) dlclose(plugin.second.library);
}

int checkRegistry(const char* path, const char* libraries, bool training) {
  using namespace cocclCompressorInternal;
  cocclConfig config;
  std::string error;
  if (!cocclLoadConfigFile(path, &config, &error)) return 1;
  config.plugins.libraryPath = libraries;
  auto& disabled = config.normal.allGather.scopes[
      static_cast<size_t>(cocclCompressionScope::Intra)];
  disabled.configured = true;
  disabled.enabled = false;
  if (training) {
    config.runtime.mode = cocclRuntimeMode::Training;
    config.trainingPolicies.dataParallel = config.normal;
    config.trainingPolicies.tensorParallel = config.normal;
    config.trainingPolicies.pipelineSendRecvForward = config.normal.sendRecv;
    config.trainingPolicies.pipelineSendRecvBackward = config.normal.sendRecv;
  }
  Registry registry;
  const cocclCompressorConfigContext context = {
      cocclCompressorConfigDefault, 2, 4};
  registrations.clear();
  if (initializeRegistry(&registry, config, context) != ncclSuccess) return 1;
  auto lookup = [&](cocclTrainingRole role, cocclOperation operation,
                    cocclCompressionScope scope,
                    cocclPolicyVariant variant = cocclPolicyVariant::Default) {
    return registry.policies[role][static_cast<size_t>(variant)]
        [static_cast<size_t>(operation)][static_cast<size_t>(scope)];
  };
  const auto role = training ? cocclTrainingRoleDataParallel : cocclTrainingRoleUnknown;
  auto* gather = lookup(role, cocclOperation::AllGather, cocclCompressionScope::Default);
  bool valid = registry.hasPolicies && gather != nullptr &&
      registry.ownedPolicies.front().get() == gather &&
      gather->thresholdBytes == config.normal.allGather.thresholdBytes &&
      lookup(role, cocclOperation::AllGather, cocclCompressionScope::Intra) == nullptr &&
      lookup(role, cocclOperation::AllGather, cocclCompressionScope::Inter) == gather &&
      registrations.size() == (training ? 16u : 8u) &&
      registry.ownedPolicies.size() == (training ? 16u : 9u);
  // A disabled intra slot must not suppress the later inherited inter registration.
  valid = valid && registrations[0].first == gather &&
      registrations[0].second == cocclDefaultPolicy(cocclOperation::AllGather) &&
      registrations[1].first == gather &&
      registrations[1].second == cocclDefaultPolicy(
          cocclOperation::AllGather, cocclCompressionScope::Inter);
  if (training) {
    valid = valid &&
        lookup(cocclTrainingRoleUnknown, cocclOperation::AllGather, cocclCompressionScope::Default) == nullptr &&
        registry.ownedPolicies[7].get() == lookup(
            cocclTrainingRoleTensorParallel, cocclOperation::AllGather, cocclCompressionScope::Default) &&
        registry.ownedPolicies[14].get() == lookup(
            cocclTrainingRolePipelineParallel, cocclOperation::SendRecv,
            cocclCompressionScope::Default, cocclPolicyVariant::Forward) &&
        registry.ownedPolicies[15].get() == lookup(
            cocclTrainingRolePipelineParallel, cocclOperation::SendRecv,
            cocclCompressionScope::Default, cocclPolicyVariant::Backward);
  } else {
    valid = valid && registry.ownedPolicies[7].get() == lookup(
        role, cocclOperation::AllToAll, cocclCompressionScope::Default) &&
        registry.ownedPolicies[8].get() == lookup(
            role, cocclOperation::SendRecv, cocclCompressionScope::Default);
  }
  closeRegistry(&registry);
  if (!valid) fprintf(stderr, "registry changed alias, role, threshold or installation order\n");
  return valid ? 0 : 1;
}

int checkRegistryErrors(const char* path, const char* libraries) {
  using namespace cocclCompressorInternal;
  cocclConfig config;
  std::string error;
  if (!cocclLoadConfigFile(path, &config, &error)) return 1;
  config.plugins.libraryPath = libraries;
  config.normal.allGather.scopes[
      static_cast<size_t>(cocclCompressionScope::Default)].values["quantBits"] = "3";
  const cocclCompressorConfigContext context = {
      cocclCompressorConfigDefault, 2, 4};
  Registry rejectedConfig;
  if (initializeRegistry(&rejectedConfig, config, context) != ncclInvalidArgument) {
    fprintf(stderr, "invalid plugin config changed its error classification\n");
    closeRegistry(&rejectedConfig);
    return 1;
  }
  closeRegistry(&rejectedConfig);
  config.plugins.libraryPath += "/missing";
  Registry missingLibrary;
  if (initializeRegistry(&missingLibrary, config, context) != ncclSystemError) {
    fprintf(stderr, "missing plugin changed its error classification\n");
    closeRegistry(&missingLibrary);
    return 1;
  }
  return 0;
}

}  // namespace

thread_local int ncclDebugNoWarn = 0;
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {}

ncclResult_t cocclAutotuneRegisterEnabledCompressor(void* compressor, cocclPolicyKey policy) {
  registrations.push_back({compressor, policy});
  return ncclSuccess;
}

int main(int argc, char** argv) {
  if (argc != 4 || checkSdp(argv[1]) || checkZfp(argv[2]) ||
      checkRegistry(argv[1], argv[3], false) ||
      checkRegistry(argv[1], argv[3], true) ||
      checkRegistryErrors(argv[1], argv[3])) return 1;
  printf("COCCL TOML policy and registry tests passed\n");
  return 0;
}
