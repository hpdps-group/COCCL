#ifndef NCCL_COMPRESSOR_UTILS_H_
#define NCCL_COMPRESSOR_UTILS_H_

#include <fstream>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>

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



inline void parseCompList(const char* env, char*** elems, int* cnt) {
    *cnt = 0;
    for(int i = 0; env[i]!='\0';i++){
        if(env[i]==',') (*cnt)++;
    }
    (*cnt)++;
    (*elems) = (char**) malloc(sizeof(char*) * (*cnt));
    char* tokStr = strdup(env);
    char* tmpStr;
    char* token = strtok_r(tokStr, ",", &tmpStr);
    int tmpIdx = 0;
    while (token) {
    //   printf("token: %s\n", token);
      (*elems)[tmpIdx++] = strdup(token);
      token = strtok_r(NULL, ",", &tmpStr);
    }
    free(tokStr);
}

inline void* tryOpenCompressorLib(const char* name) {
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