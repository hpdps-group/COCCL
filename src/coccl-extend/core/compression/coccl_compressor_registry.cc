#include "core/compression/coccl_compressor_internal.h"
#include "core/compression/coccl_compressor_config.h"

#include "core/tuning/coccl_autotune.h"
#include "checks.h"
#include "debug.h"

#include <dlfcn.h>
#include <utility>

namespace cocclCompressorInternal {
namespace {

ncclResult_t loadPlugins(Registry* registry, const cocclConfig& config) {
  for (const std::string& name : config.plugins.compressors) {
    const std::string path = cocclCompressorPluginPath(config, name);
    void* library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      WARN("COCCL failed to open compressor %s: %s", path.c_str(), dlerror());
      return ncclSystemError;
    }
    auto entry = reinterpret_cast<cocclGetCompressorPluginFn>(
        dlsym(library, COCCL_COMPRESSOR_ENTRY_SYMBOL));
    const cocclCompressorPlugin* plugin = entry == nullptr ? nullptr : entry();
    char error[192] = {};
    if (!cocclValidateCompressorPlugin(name.c_str(), plugin, error,
                                       sizeof(error))) {
      WARN("COCCL compressor %s rejected: %s", name.c_str(), error);
      dlclose(library);
      return ncclInvalidArgument;
    }
    registry->loadedPlugins.emplace(name, LoadedPlugin{library, plugin});
  }
  return ncclSuccess;
}

ncclResult_t createPolicy(
    Registry* registry, const cocclCompressorScopeEntry& configured,
    const cocclCompressorConfigContext& context, CompressorPolicy** policy) {
  auto plugin = registry->loadedPlugins.find(configured.name);
  if (plugin == registry->loadedPlugins.end()) return ncclInvalidArgument;

  cocclCompressorConfigViewStorage storage(configured.values);
  const cocclConfigView view = storage.view();
  char error[256] = {};
  void* parsedConfig = nullptr;
  ncclResult_t result = plugin->second.descriptor->parseConfig(
      &view, &context, &parsedConfig, error, sizeof(error));
  if (result != ncclSuccess) {
    WARN("COCCL compressor %s configuration rejected: %s",
         configured.name.c_str(), error);
    return result;
  }

  auto created = std::make_unique<CompressorPolicy>();
  created->plugin = plugin->second.descriptor;
  created->config = parsedConfig;
  *policy = created.get();
  registry->ownedPolicies.push_back(std::move(created));
  return ncclSuccess;
}

ncclResult_t installPolicy(Registry* registry, cocclTrainingRole trainingRole,
                           cocclOperation operation,
                           const cocclPrimitivePolicy& configured,
                           cocclPolicyVariant variant,
                           const cocclCompressorConfigContext& context) {
  const size_t index = static_cast<size_t>(operation);
  const size_t role = static_cast<size_t>(trainingRole);
  const size_t policyVariant = static_cast<size_t>(variant);
  for (cocclCompressionScope scope : {
           cocclCompressionScope::Default,
           cocclCompressionScope::Intra,
           cocclCompressionScope::Inter}) {
    const size_t scopeIndex = static_cast<size_t>(scope);
    const cocclEffectiveCompressorScope effective =
        cocclEffectiveCompressorScopeFor(configured, scope);
    if (!effective.enabled()) continue;
    registry->hasPolicies = true;
    if (scope != cocclCompressionScope::Default &&
        effective.source == cocclCompressionScope::Default) {
      registry->policies[role][policyVariant][index][scopeIndex] =
          registry->policies[role][policyVariant][index][static_cast<size_t>(
              cocclCompressionScope::Default)];
      continue;
    }
    NCCLCHECK(createPolicy(
        registry, *effective.entry, context,
        &registry->policies[role][policyVariant][index][scopeIndex]));
    registry->policies[role][policyVariant][index][scopeIndex]->thresholdBytes =
        configured.thresholdBytes;
  }
  return ncclSuccess;
}

cocclTrainingRole trainingRole(cocclPolicyScope scope) {
  switch (scope) {
    case cocclPolicyScope::Normal: return cocclTrainingRoleUnknown;
    case cocclPolicyScope::DataParallel: return cocclTrainingRoleDataParallel;
    case cocclPolicyScope::TensorParallel: return cocclTrainingRoleTensorParallel;
    case cocclPolicyScope::PipelineParallel: return cocclTrainingRolePipelineParallel;
  }
  __builtin_unreachable();
}

}  // namespace

ncclResult_t initializeRegistry(
    Registry* registry, const cocclConfig& config,
    const cocclCompressorConfigContext& context) {
  NCCLCHECK(loadPlugins(registry, config));
  for (const cocclConfigPolicyView& binding : cocclEnumeratePolicies(config)) {
    if (binding.mode != config.runtime.mode) continue;
    NCCLCHECK(installPolicy(
        registry, trainingRole(binding.scope), binding.key.operation,
        *binding.policy, binding.key.variant, context));
  }

  for (size_t role = 0; role < kTrainingRoleCount; ++role) {
    for (cocclOperation operation : {
             cocclOperation::AllGather, cocclOperation::ReduceScatter,
             cocclOperation::AllReduce}) {
      CompressorPolicy* previous = nullptr;
      for (cocclCompressionScope scope : {
               cocclCompressionScope::Default,
               cocclCompressionScope::Intra,
               cocclCompressionScope::Inter}) {
        CompressorPolicy* policy =
            registry->policies[role][static_cast<size_t>(cocclPolicyVariant::Default)]
                    [static_cast<size_t>(operation)]
                    [static_cast<size_t>(scope)];
        if (policy != nullptr && policy != previous) {
          NCCLCHECK(cocclAutotuneRegisterEnabledCompressor(
              policy, cocclDefaultPolicy(operation, scope)));
        }
        previous = policy;
      }
    }
  }
  return ncclSuccess;
}

}  // namespace cocclCompressorInternal
