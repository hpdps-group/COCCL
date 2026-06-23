export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=/lib/x86_64-linux-gnu/libtinfo.so.6:$LD_PRELOAD


# source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source ~/.bashrc
# conda activate /mnt/afs/lm/envs/pai-h100


which python || echo "Critical Error: Python not found!"



export NCCL_HOME=/mnt/afs/lm/lm/coccl-acc/build
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$NCCL_HOME/lib/libnccl.so:$LD_PRELOAD 

# export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=/mnt/afs/lm/lm/coccl-training/logs/qwen2.5/7B/nccl_debug.log

export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker,veth

export NCCL_ENABLE_COMPRESS=1
export NCCL_COMPRESSORS=taco
export NCCL_ENABLE_ALLREDUCE_COMPRESS=1
export NCCL_ALLREDUCE_COMPRESSORS=taco
export NCCL_ENABLE_ALLGATHER_COMPRESS=0
export NCCL_ALLGATHER_COMPRESSORS=taco
export NCCL_ENABLE_REDUCESCATTER_COMPRESS=0
export NCCL_REDUCESCATTER_COMPRESSORS=taco
export NCCL_REDUCESCATTER_INTER_COMPRESSORS=taco
export NCCL_ENABLE_SENDRECV_COMPRESS=0
export NCCL_SENDRECV_COMPRESSORS=taco
export NCCL_SENDRECV_BWD_COMPRESSORS=taco

export NCCL_COMPRESSORS_CONFIG_PATH=/mnt/afs/lm/lm/coccl-acc/src/device/compress/configs
export NCCL_COMPRESSORS_LIB_PATH=/mnt/afs/lm/lm/coccl-acc/build/obj/device/compress/libcompress


GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0


cd /mnt/afs/lm/Pai-Megatron-Patch/examples/qwen2_5
unset MP_DATASET_TYPE 



sh run_mcore_qwen.sh  \
dsw  \
0.5B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
bf16  \
1   \
1  \
1 \
true \
true   \
true \
true \
false   \
false \
1000  \
/mnt/afs/lm/lm/TACO-COCCL/mnt/qwen-datasets/wudao_qwenbpe_text_document   \
/mnt/afs/lm/lm/TACO-COCCL/mnt/qwen-datasets/wudao_qwenbpe_text_document   \
/mnt/afs/lm/lm/TACO-COCCL/mnt/qwen-ckpts/Qwen2.5-0.5B-to-mcore  \
1000  \
10   \
/mnt/afs/lm/lm/TACO-COCCL/logs/qwen2.5_0.5B    2>&1 | tee  /mnt/afs/lm/lm/TACO-COCCL/logs/qwen2.5-0.5B-taco.log