#ifndef NCCL_COMPRESSOR_UTILS_H_
#define NCCL_COMPRESSOR_UTILS_H_

#include <cerrno>
#include <climits>
#include <cstdio>
#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

// Parses "key: value" config lines while trimming leading whitespace. This is a
// macro because the caller owns the local line buffer and parse loop.
#define getConfigLinePair()                               \
          char* saveptr;                                  \
          char* feature = strtok_r(line, ":", &saveptr);  \
          if(!feature) continue;                          \
          char* value = strtok_r(NULL, "", &saveptr);     \
          if(!value) continue;                            \
          char* kk;                                       \
          feature = strtok_r(feature, " \t",&kk);         \
          char* vv;                                       \
          value = strtok_r(value, " \t",&vv);             \
          if(!feature || !value) continue;                \


inline void loadConfigPair(const char* configFile, std::pair<const char*, const char*>** configPairs, int* configPairCount){
    // Two-pass parser: count first so legacy compressor config structs can keep
    // a compact malloc-owned array of key/value pairs.
    std::ifstream filecount(configFile);
    *configPairCount = 0;
    if(filecount.is_open()){
        char line[1024];
        while(filecount.getline(line, 1024)){
            getConfigLinePair()
            *configPairCount += 1;
        }
    }
    if(*configPairs == NULL || *configPairs == nullptr)
        *configPairs = (std::pair<const char*, const char*>*)
                        malloc(sizeof(std::pair<const char*, const char*>) * (*configPairCount));
    std::ifstream file(configFile);
    int idx = 0;
    if(file.is_open()){
        char line[1024];
        while(file.getline(line, 1024)){
            getConfigLinePair()
            (*configPairs)[idx++] = std::make_pair(strdup(feature), strdup(value));
        }
    }
}

// NCCL_COMPRESSORS-style lists are comma-separated compressor symbol names.
// Empty entries are ignored so "a,,b," resolves to {"a", "b"}.
inline std::vector<std::string> parseCompressorNames(const char* env) {
    std::vector<std::string> names;
    if (env == nullptr || env[0] == '\0') return names;

    const char* start = env;
    for (const char* p = env;; ++p) {
        if (*p == ',' || *p == '\0') {
            if (p > start) names.emplace_back(start, p - start);
            if (*p == '\0') break;
            start = p + 1;
        }
    }
    return names;
}

inline std::string buildCompressorLibPath(const char* libPath, const std::string& compName) {
    char compLibName[PATH_MAX];
    snprintf(compLibName, PATH_MAX, "%s/lib%s.so", libPath != nullptr ? libPath : "", compName.c_str());
    return std::string(compLibName);
}

inline std::string buildCompressorConfigPath(const char* configPathBase, const std::string& compName,
                                             const char* configSuffix) {
    char compConfigPath[PATH_MAX];
    snprintf(compConfigPath, PATH_MAX, "%s/%s/%s_%s.config",
             configPathBase != nullptr ? configPathBase : "", compName.c_str(), compName.c_str(), configSuffix);
    return std::string(compConfigPath);
}

inline bool compressorConfigFileExists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

inline void* tryOpenCompressorLib(const char* name) {
    // Runtime treats missing/empty names as "not found"; warning policy lives
    // at the caller so it can include the env variable context.
    if (nullptr == name || strlen(name) == 0) {
      return nullptr;
    }
    void *handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
    if (nullptr == handle) {
      if (ENOENT == errno) {
        // INFO(NCCL_ENV, "Compressor/Lib: No library found (%s)", name);
      }
    }
    return handle;
}


#endif
