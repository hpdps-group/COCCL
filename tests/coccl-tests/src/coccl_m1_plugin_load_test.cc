#include "compressor_plugin/detail/coccl_compressor_abi.h"

#include <dlfcn.h>
#include <stdio.h>

#include <string>

int checkPlugin(const char* directory, const char* name) {
  const std::string path =
      std::string(directory) + "/lib" + name + ".so";
  void* library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr) {
    fprintf(stderr, "failed to load %s: %s\n", path.c_str(), dlerror());
    return 1;
  }
  auto entry = reinterpret_cast<cocclGetCompressorPluginFn>(
      dlsym(library, COCCL_COMPRESSOR_ENTRY_SYMBOL));
  const cocclCompressorPlugin* plugin = entry == nullptr ? nullptr : entry();
  char error[160] = {};
  if (!cocclValidateCompressorPlugin(name, plugin, error, sizeof(error))) {
    fprintf(stderr, "%s descriptor rejected: %s\n", name, error);
    dlclose(library);
    return 1;
  }
  if (std::string(name) == "sdp4bit" &&
      (plugin->capabilities &
       cocclCompressorCapabilityFusedHierarchicalSwizzle) == 0) {
    fprintf(stderr, "sdp4bit is missing fused hierarchical swizzle\n");
    dlclose(library);
    return 1;
  }
  if ((std::string(name) == "zfp" || std::string(name) == "dietgpu") &&
      (plugin->capabilities &
       cocclCompressorCapabilityFusedHierarchicalSwizzle) != 0) {
    fprintf(stderr, "%s unexpectedly advertises fused hierarchical swizzle\n",
            name);
    dlclose(library);
    return 1;
  }
  cocclCompressorPlugin legacy = *plugin;
  legacy.abiVersion = 7;
  if (cocclValidateCompressorPlugin(name, &legacy, error, sizeof(error))) {
    fprintf(stderr, "%s accepted an ABI v7 descriptor\n", name);
    dlclose(library);
    return 1;
  }
  dlclose(library);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2 || checkPlugin(argv[1], "sdp4bit") ||
      checkPlugin(argv[1], "zfp")) {
    return 1;
  }
  for (int index = 2; index < argc; ++index) {
    if (checkPlugin(argv[1], argv[index])) return 1;
  }
  printf("COCCL M1 plugin load tests passed\n");
  return 0;
}
