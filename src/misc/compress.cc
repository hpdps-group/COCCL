
#include "compress.h"
#include "info.h"
#include "align.h"
#include "param.h"
#include "comm.h"
#include <stdlib.h>
#include <mutex>
#include <thread>
#include "coccl_alloc.h"
#include "reduce_extend.h"

extern "C"{
ncclResult_t launchCompress(const void* orgbuff, const size_t orgChunkCount, ncclDataType_t orgType,  void* compbuff, 
    const size_t compChunkCount, ncclDataType_t compType, const size_t numChunks, cudaStream_t stream);

ncclResult_t launchDecompress(const void* compbuff, const size_t compChunkCount, ncclDataType_t compType, void *decompbuff, 
    const size_t decompChunkCount, ncclDataType_t decompType, const size_t numChunks, cudaStream_t stream);

ncclResult_t launchDecompressReduce(const void* compbuff, const size_t compChunkCount, ncclDataType_t compType, const void* reduceInput, void *output, 
    const size_t OutputChunkCount, ncclDataType_t OutputType, const size_t numChunks, cudaStream_t stream);

}
// NCCL_PARAM(CompressEnable, "COMPRESS_ENABLE", 0);

pthread_mutex_t compressorLibLock = PTHREAD_MUTEX_INITIALIZER;

// Init compress memory pool
cudaMemPool_t* compMemPool = NULL;
size_t compMemPoolCnt = 1; 
NCCL_PARAM(CompMemPoolHoldSize, "COMPRESS_MEMPOOL_HOLDSIZE", 4 * 1024 * 1024);

ncclResult_t initCompMemPool(const ncclComm_t comm){
    INFO(NCCL_INIT, "Init Compress Memory Pool");
    
    pthread_mutex_lock(&compressorLibLock);
    if(compMemPool == NULL){
        // printf("threadId %d initCount %d rank %d localDev %d\n", std::this_thread::get_id(), ++cnt, commRank, localDev);
        compMemPool = (cudaMemPool_t*) calloc(comm->intraRanks, sizeof(cudaMemPool_t));
        compMemPoolCnt = comm->intraRanks;
    }

    if(compMemPool != NULL && compMemPool[comm->cudaDev % comm->intraRanks] == NULL){

        cuuint64_t CompMemPoolHoldSize = (cuuint64_t) ncclParamCompMemPoolHoldSize();
        cudaSetDevice(comm->cudaDev);
        cudaMemPoolProps poolProps = { };
        poolProps.allocType = cudaMemAllocationTypePinned;
        poolProps.location.id = comm->cudaDev;
        poolProps.location.type = cudaMemLocationTypeDevice;
        CUDACHECK(cudaMemPoolCreate(&compMemPool[comm->cudaDev % comm->intraRanks], &poolProps));
        
        // usually it is 8 devices
        int i = 0;
        for(ncclComm* c = comm->intraComm0; c != NULL && i<comm->intraRanks; c=c->intraNext, i++){
            int peer = c->cudaDev;
            if(comm->cudaDev == peer) continue;
            cudaMemAccessDesc accessDesc = {};
            accessDesc.location.type = cudaMemLocationTypeDevice;
            accessDesc.location.id = peer;
            accessDesc.flags = cudaMemAccessFlagsProtReadWrite;
            int canAccess = 0;
            CUDACHECK(cudaDeviceCanAccessPeer(&canAccess, peer, comm->cudaDev));
            if (canAccess) {
                CUDACHECK(cudaMemPoolSetAccess(compMemPool[comm->cudaDev % comm->intraRanks], &accessDesc, 1));
            }
        }
        CUDACHECK(cudaMemPoolSetAttribute(compMemPool[comm->cudaDev % comm->intraRanks], cudaMemPoolAttrReleaseThreshold, &CompMemPoolHoldSize));
    }
    pthread_mutex_unlock(&compressorLibLock);

    return ncclSuccess;
}

struct ncclCompElem {
    ncclCompressor_t* compressor = nullptr;
    void* compConfig = nullptr;
    ncclCompElem* next = nullptr;
    ncclCompElem* prev = nullptr;
};
struct ncclCompList {
    ncclCompElem* head = nullptr;
    ncclCompElem* tail = nullptr;
};

static void compPushBack(ncclCompList* list, ncclCompElem* elem){
    if(list->tail == nullptr){
        elem->next = nullptr;
        elem->prev = nullptr;
        list->head = list->tail = elem;
    } else {
        list->tail->next = elem;
        elem->prev = list->tail;
        elem->next = nullptr;
        list->tail = elem;
    }
}

static ncclCompressor_t* findCompHandle(ncclCompList* list, const char* compName){
    ncclCompElem* elem = list->head;
    while(elem != nullptr){
        if(strcmp(elem->compressor->name, compName) == 0){
            return elem->compressor;
        }
        elem = elem->next;
    }
    return nullptr;
}

// all used compressors
static ncclCompList compList;
// AlltoAll used compressors. Default is for Intra- and Inter-(if Inter is null)
static ncclCompList compListA2A;
// Inter is only for inter-node comm
static ncclCompList compListA2AInter;
// AllReduce
static ncclCompList compListAR;
static ncclCompList compListARInter;
// AllGather
static ncclCompList compListAG;
static ncclCompList compListAGInter;
// ReduceScatter
static ncclCompList compListRS;
static ncclCompList compListRSInter;
// SendRecv
static ncclCompList compListSR;
static ncclCompList compListSRBWD;

#define COMPRESSLIB_PATH(libpath, libname) \
    snprintf(compLibName, PATH_MAX, "%s/lib%s.so", libpath, libname)

#define COMPRESSCONFIG_PATH(compName, ALGO, ...)                                                                                    \
    if(sizeof(#__VA_ARGS__) == 1)                                                                                                   \
        snprintf(compConfigPath, PATH_MAX, "%s/%s/%s_%s.config", compConfigPathBase, compName, compName, #ALGO);                    \
    else                                                                                                                            \
        snprintf(compConfigPath, PATH_MAX, "%s/%s/%s_%s_%s.config", compConfigPathBase, compName, compName, #ALGO, #__VA_ARGS__)

#define LOADCOMPRESSOR(env, ALGO, ...)                                                                      \
    usedComp = getenv(env);                                                                                 \
    if(usedComp != NULL){                                                                                   \
        parseCompList(usedComp, &compName, &numComp);                                                       \
        for(int i = 0; i < numComp; i++){                                                                   \
            ncclCompElem* compElem = (ncclCompElem*) malloc(sizeof(ncclCompElem));                          \
            compElem->compressor = findCompHandle(&compList, compName[i]);                                  \
            COMPRESSCONFIG_PATH(compName[i], ALGO, ##__VA_ARGS__);                                          \
            compElem->compressor->parseConfig(compConfigPath, &compElem->compConfig, comm->nNodes, comm->localRanks);  \
            compElem->next = nullptr;                                                                       \
            compElem->prev = nullptr;                                                                       \
            compPushBack(&compList##ALGO##__VA_ARGS__, compElem);                                           \
        }                                                                                                   \
        for (int i = 0; i < numComp; i++) free(compName[i]);                                                \
        free(compName);                                                                                     \
    } else {                                                                                                \
        compList##ALGO##__VA_ARGS__ = compList##ALGO;                                                       \
    }                                                                                                       


bool enableAllToAllComp = false;
bool enableAllReduceComp = false;
bool enableAllGatherComp = false;
bool enableReduceScatterComp = false;
bool enableSendRecvComp = false;
bool enableCheck = false;
bool enableTimer = false;
size_t CompEnableThreshold = 0;
NCCL_PARAM(CompEnableThreshold, "COMPRESS_ENABLE_THRESHOLD", 0);


static void loadCompressors(const ncclComm_t comm) {
    // Load AllCompressors
    pthread_mutex_lock(&compressorLibLock);
    INFO(NCCL_INIT, "Load Compressors");
    char** compName = nullptr;
    int numComp=0;
    const char* usedComp = getenv("NCCL_COMPRESSORS");
    char compLibName[PATH_MAX];
    parseCompList(usedComp, &compName, &numComp);
    const char* compLibPath = getenv("NCCL_COMPRESSORS_LIB_PATH");
    for(int i = 0; i < numComp; i++){        
        // printf("load Compressos %s\n", compName[i]);
        COMPRESSLIB_PATH(compLibPath, compName[i]);         
        void* compLibHandle = tryOpenCompressorLib(compLibName);  
        ncclCompElem* compElem = (ncclCompElem*) malloc(sizeof(ncclCompElem));   
        compElem->compressor = (ncclCompressor_t*) dlsym(compLibHandle, compName[i]); 
        // get defualt config
        compElem->compressor->parseConfig(nullptr, &compElem->compConfig, comm->nNodes, comm->localRanks);
        compElem->compConfig = nullptr;                                                     
        compElem->next = nullptr;                                                           
        compElem->prev = nullptr;                                                           
        compPushBack(&compList, compElem);
    }                                                                                       
    free(compName);
    const char* compConfigPathBase = getenv("NCCL_COMPRESSORS_CONFIG_PATH");
    char compConfigPath[PATH_MAX];

    
    const char* usedAllToAllComp = getenv("NCCL_ENABLE_ALLTOALL_COMPRESS");
    if(usedAllToAllComp && strcmp(usedAllToAllComp, "1") == 0){
        enableAllToAllComp = true;
        LOADCOMPRESSOR("NCCL_ALLTOALL_COMPRESSORS", A2A);
        LOADCOMPRESSOR("NCCL_ALLTOALL_INTER_COMPRESSORS", A2A, Inter);
    }
    const char* usedAllReduceComp = getenv("NCCL_ENABLE_ALLREDUCE_COMPRESS");
    if(usedAllReduceComp && strcmp(usedAllReduceComp, "1") == 0){
        enableAllReduceComp = true;
        LOADCOMPRESSOR("NCCL_ALLREDUCE_COMPRESSORS", AR);
        LOADCOMPRESSOR("NCCL_ALLREDUCE_INTER_COMPRESSORS", AR, Inter);
    }
    const char* usedAllGatherComp = getenv("NCCL_ENABLE_ALLGATHER_COMPRESS");
    if(usedAllGatherComp && strcmp(usedAllGatherComp, "1") == 0){
        enableAllGatherComp = true;
        LOADCOMPRESSOR("NCCL_ALLGATHER_COMPRESSORS", AG);
        LOADCOMPRESSOR("NCCL_ALLGATHER_INTER_COMPRESSORS", AG, Inter);
    }
    const char* usedReduceScatterComp = getenv("NCCL_ENABLE_REDUCESCATTER_COMPRESS");
    if(usedReduceScatterComp && strcmp(usedReduceScatterComp, "1") == 0){
        enableReduceScatterComp = true;
        LOADCOMPRESSOR("NCCL_REDUCESCATTER_COMPRESSORS", RS);
        LOADCOMPRESSOR("NCCL_REDUCESCATTER_INTER_COMPRESSORS", RS, Inter);
    }
    const char* usedSendRecvComp = getenv("NCCL_ENABLE_SENDRECV_COMPRESS");
    if(usedSendRecvComp && strcmp(usedSendRecvComp, "1") == 0){
        enableSendRecvComp = true;
        LOADCOMPRESSOR("NCCL_SENDRECV_COMPRESSORS", SR);
        LOADCOMPRESSOR("NCCL_SENDRECV_BWD_COMPRESSORS", SR, BWD);
    }
    const char* isCheck = getenv("NCCL_ENABLE_CHECK");
    if(isCheck && strcmp(isCheck, "1") == 0){
        enableCheck = true;
    }
    const char* isTimer = getenv("NCCL_ENABLE_TIMER");
    if(isTimer && strcmp(isTimer, "1") == 0){
        enableTimer = true;
    }
    CompEnableThreshold = (size_t) ncclParamCompEnableThreshold();

    pthread_mutex_unlock(&compressorLibLock);

} 

static ncclResult_t initCompressors(const ncclComm_t comm){
    if(compList.head == nullptr){
        loadCompressors(comm);
    }
    return ncclSuccess;
}


ncclResult_t ncclCompressInit(const ncclComm_t comm){
    NCCLCHECK(initCompMemPool(comm));
    NCCLCHECK(initCompressors(comm));
    return ncclSuccess;
}

#define DOCOMPRESS(ALGO, ...)                                                                                                 \
    for(ncclCompElem* compElem = (compList##ALGO##__VA_ARGS__).head; compElem != nullptr; compElem = compElem->next){         \
        CUDACHECK(compElem->compressor->compress(orgbuff, compbuff, orgChunkCount, orgDayatype,                               \
            compChunkCount, compDatatype, numChunks, rank, compElem->compConfig, NULL, stream));   \
    }
// compress
ncclResult_t ncclCompress(const void* orgbuff, void** compbuff, const size_t orgChunkCount, ncclDataType_t orgDayatype,
    size_t* compChunkCount, ncclDataType_t* compDatatype, const size_t numChunks, const int rank, ncclCommOp_t commOp, cudaStream_t stream)
{
    int cudaDev;
    CUDACHECK(cudaGetDevice(&cudaDev));
//    printf("comm Dev %d\n", cudaDev);
    switch(commOp){
        case ncclCommOp_t::AlltoAll:
            DOCOMPRESS(A2A)
            break;
        case ncclCommOp_t::AlltoAll_Inter:
            DOCOMPRESS(A2A, Inter)
            break;
        case ncclCommOp_t::AllReduce:
            DOCOMPRESS(AR)
            break;
        case ncclCommOp_t::AllReduce_Inter:
            DOCOMPRESS(AR, Inter)
            break;
        case ncclCommOp_t::AllGather:
            DOCOMPRESS(AG)
            break;
        case ncclCommOp_t::AllGather_Inter:
            DOCOMPRESS(AG, Inter)
            break;
        case ncclCommOp_t::ReduceScatter:
            DOCOMPRESS(RS)
            break;
        case ncclCommOp_t::ReduceScatter_Inter:
            DOCOMPRESS(RS, Inter)
            break;
        case ncclCommOp_t::SendRecv:
            DOCOMPRESS(SR)
            break;
        case ncclCommOp_t::SendRecv_BWD:
            DOCOMPRESS(SR, BWD)
            break;
        default:
            DOCOMPRESS()
    }
    return ncclSuccess;
}

#define DODECOMPRESS(ALGO, ...)                                                                                           \
    for(ncclCompElem* compElem = (compList##ALGO##__VA_ARGS__).tail; compElem != nullptr; compElem = compElem->prev){     \
        CUDACHECK(compElem->compressor->decompress(decompbuff, compbuff, decompChunkCount, decompDatatype,                \
            compChunkCount, compDatatype, numChunks, compElem->compConfig, stream));                                      \
    }
ncclResult_t ncclDecompress(void* decompbuff, const void* compbuff, const size_t decompChunkCount, ncclDataType_t decompDatatype,
    const size_t compChunkCount, ncclDataType_t compDatatype, const size_t numChunks, ncclCommOp_t commOp, cudaStream_t stream)
{
    switch(commOp){
        case ncclCommOp_t::AlltoAll:
            DODECOMPRESS(A2A)
            break;
        case ncclCommOp_t::AlltoAll_Inter:
            DODECOMPRESS(A2A, Inter)
            break;
        case ncclCommOp_t::AllReduce:
            DODECOMPRESS(AR)
            break;
        case ncclCommOp_t::AllReduce_Inter:
            DODECOMPRESS(AR, Inter)
            break;
        case ncclCommOp_t::AllGather:
            DODECOMPRESS(AG)
            break;
        case ncclCommOp_t::AllGather_Inter:
            DODECOMPRESS(AG, Inter)
            break;
        case ncclCommOp_t::ReduceScatter:
            DODECOMPRESS(RS)
            break;
        case ncclCommOp_t::ReduceScatter_Inter:
            DODECOMPRESS(RS, Inter)
            break;
         case ncclCommOp_t::SendRecv:
            DODECOMPRESS(SR)
            break;
        case ncclCommOp_t::SendRecv_BWD:
            DODECOMPRESS(SR, BWD)
            break;
        default:
            DODECOMPRESS()
    }

    return ncclSuccess;
}

__thread void* reduceTempbuff = nullptr;
ncclResult_t cocclDecompReduce(void* reducebuff, const void* compbuff, const size_t compChunkCount, ncclDataType_t compDatatype, 
    const size_t reduceChunkCount, ncclDataType_t reduceDataType, const size_t numChunks, ncclCompElem* compElem, cudaStream_t stream)
{
    NCCLCHECK(cocclBuffAlloc(&reduceTempbuff, reduceChunkCount * numChunks * ncclTypeSize(reduceDataType), nullptr)); 

    CUDACHECK(compElem->compressor->decompress(reduceTempbuff, compbuff, reduceChunkCount, reduceDataType, 
        compChunkCount, compDatatype, numChunks, compElem->compConfig, stream));
    
    NCCLCHECK(ncclReduceChunk(reduceTempbuff, reduceChunkCount, reducebuff, reduceDataType, numChunks, stream));

    return ncclSuccess;
}

#define DODECOMPREDUCE(ALGO, ...)                                                                                           \
    for(ncclCompElem* compElem = (compList##ALGO##__VA_ARGS__).tail; compElem != nullptr; compElem = compElem->prev){       \
        if(compElem->compressor->decompReduce == nullptr) {                                                                 \
            NCCLCHECK(cocclDecompReduce(reducebuff, compbuff, compChunkCount, compDatatype, reduceChunkCount,               \
                reduceDataType, numChunks, compElem, stream));                                                              \
        }else {                                                                                                             \
            CUDACHECK(compElem->compressor->decompReduce(reducebuff, compbuff, compChunkCount, compDatatype,                \
                reduceChunkCount, reduceDataType, numChunks, compElem->compConfig, stream));                                \
        }                                                                                                                   \
    }
ncclResult_t ncclDecompressReduce(void* reducebuff, const void* compbuff, const size_t compChunkCount, ncclDataType_t compDatatype, 
    const size_t reduceChunkCount, ncclDataType_t reduceDataType, const size_t numChunks, ncclCommOp_t commOp, cudaStream_t stream)
{
    switch(commOp){
        case ncclCommOp_t::AllReduce:
            DODECOMPREDUCE(AR)
            break;
        case ncclCommOp_t::AllReduce_Inter:
            DODECOMPREDUCE(AR, Inter)
            break;
        case ncclCommOp_t::ReduceScatter:
            DODECOMPREDUCE(RS)
            break;
        case ncclCommOp_t::ReduceScatter_Inter:
            DODECOMPREDUCE(RS, Inter)
            break;
        default:
            DODECOMPREDUCE()
    }
    return ncclSuccess;
}

ncclResult_t cocclDecompReduceComp(const void* compbuff, void** recompbuff, const size_t orgChunkCount, ncclDataType_t orgDayatype,
    const size_t compChunkCount, ncclDataType_t compDatatype, size_t* reCompChunkCount, ncclDataType_t* reCompDatatype, const size_t numChunks, 
    ncclCompElem* compElem, cudaStream_t stream)
{
    NCCLCHECK(cocclBuffAlloc(&reduceTempbuff, orgChunkCount * numChunks * ncclTypeSize(orgDayatype), nullptr)); 

    CUDACHECK(compElem->compressor->decompress(reduceTempbuff, compbuff, orgChunkCount, orgDayatype, 
        compChunkCount, compDatatype, numChunks, compElem->compConfig, stream));
    
    NCCLCHECK(ncclReduceChunk(reduceTempbuff, orgChunkCount, reduceTempbuff, orgDayatype, numChunks, stream));

    CUDACHECK(compElem->compressor->compress(reduceTempbuff, recompbuff, orgChunkCount, orgDayatype,                              
        reCompChunkCount, reCompDatatype, 1, 0, compElem->compConfig, NULL, stream)); 
    
    return ncclSuccess;
}

#define DODECOMPREDUCECOMP(ALGO, ...)                                                                                           \
    for(ncclCompElem* compElem = (compList##ALGO##__VA_ARGS__).tail; compElem != nullptr; compElem = compElem->prev){           \
        if(compElem->compressor->decompReduceComp == nullptr)   {                                                               \
            NCCLCHECK(cocclDecompReduceComp(compbuff, recompbuff, orgChunkCount, orgDayatype, compChunkCount,                   \
                compDatatype, reCompChunkCount, reCompDatatype, numChunks, compElem, stream));                                  \
        } else     {                                                                                                            \
            CUDACHECK(compElem->compressor->decompReduceComp(compbuff, recompbuff, compChunkCount, compDatatype,                \
                reCompChunkCount, reCompDatatype, numChunks, compElem->compConfig, NULL, stream));                              \
        }                                                                                                                       \
    }
ncclResult_t ncclDecompReduceComp(const void* compbuff, void** recompbuff, const size_t orgChunkCount, ncclDataType_t orgDayatype,
    const size_t compChunkCount, ncclDataType_t compDatatype, size_t* reCompChunkCount, ncclDataType_t* reCompDatatype, const size_t numChunks, 
    ncclCommOp_t commOp, cudaStream_t stream)
{
    int cudaDev;
    CUDACHECK(cudaGetDevice(&cudaDev));
    switch(commOp){
        case ncclCommOp_t::AllReduce:
            DODECOMPREDUCECOMP(AR)
            break;
        case ncclCommOp_t::AllReduce_Inter:
            DODECOMPREDUCECOMP(AR, Inter)
            break;
        case ncclCommOp_t::ReduceScatter:
            DODECOMPREDUCECOMP(RS)
            break;
        case ncclCommOp_t::ReduceScatter_Inter:
            DODECOMPREDUCECOMP(RS, Inter)
            break;
        default:
            DODECOMPREDUCECOMP()
    }
    return ncclSuccess;
}

// ncclResult_t ncclDecompressReduce(void* reducebuff, const void* reduceInput, const void* compbuff, const size_t outChunkCount, ncclDataType_t outDatatype, 
//     const size_t compChunkCount, ncclDataType_t compDatatype,  const size_t numChunks, cudaStream_t stream)
// {

//     NCCLCHECK(launchDecompressReduce(compbuff, compChunkCount, compDatatype, reduceInput, reducebuff, outChunkCount, outDatatype, numChunks, stream));
//     return ncclSuccess;
// }



