#ifndef NCCL_COMP_WRAPPER_H_
#define NCCL_COMP_WRAPPER_H_

#include "nccl.h"
#include "collectives.h"
#include "coll_extend_p2p.h"
#include "comm.h"
#include "argcheck.h"

extern __thread struct parComm* parcomms;
struct parComm{
    ncclComm_t subcomm;
    cudaStream_t stream;
    cudaEvent_t event;
};
NCCL_PARAM(PipelineDepth, "PIPELINE_DEPTH", 0);

// // init subcomm
// inline ncclResult_t initParallelComms(ncclComm_t comm) {
//     if(parcomms == nullptr) {
//         CUDACHECK(cudaSetDevice(comm->cudaDev));
//         if(pipelineSize == -1){
//             pipelineSize = ncclParamPipelineSize();
//         }
//         parcomms = (parComm*)malloc(sizeof(parComm) * (pipelineSize));
//         for(int i = 0; i < pipelineSize; i++){
//             NCCLCHECK(ncclCommSplit(comm, i, comm->rank, &parcomms[i].subcomm, NULL));
//             CUDACHECK(cudaStreamCreateWithFlags(&parcomms[i].stream, cudaStreamNonBlocking));
//             CUDACHECK(cudaEventCreateWithFlags(&parcomms[i].event, cudaEventDefault));
//         }
//     }
//     return ncclSuccess;
// }

// inline ncclResult_t freeParallelComms(ncclComm_t comm){
//     if(parcomms != nullptr){
//         for(int i = 0; i < 2; i++){
//             NCCLCHECK(ncclCommDestroy(parcomms[i].subcomm));
//             CUDACHECK(cudaStreamDestroy(parcomms[i].stream));
//             CUDACHECK(cudaEventDestroy(parcomms[i].event));
//         }
//         free(parcomms);
//     }
//     return ncclSuccess;
// }

#endif