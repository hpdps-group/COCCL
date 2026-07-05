#!/bin/bash
module load miniconda/24.9.2 cuda/12.4
source activate python3.10
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

source training_envs.sh

export NCCL_HOME=$COCCL_PATH/build
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$NCCL_HOME/lib:$LIBRARY_PATH
export C_INCLUDE_PATH=$NCCL_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$NCCL_HOME/include:$CPLUS_INCLUDE_PATH

echo "$COCCL_PATH"

export NCCL_DEBUG=WRAN
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=13

export NCCL_ENABLE_COMPRESS=1
export NCCL_COMPRESSORS=sdp4bit,tahquant
export NCCL_ENABLE_ALLGATHER_COMPRESS=1
export NCCL_ALLGATHER_COMPRESSORS=sdp4bit
export NCCL_ENABLE_REDUCESCATTER_COMPRESS=1
export NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit
export NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit
export NCCL_ENABLE_SENDRECV_COMPRESS=1
export NCCL_SENDRECV_COMPRESSORS=tahquant
export NCCL_SENDRECV_BWD_COMPRESSORS=tahquant
export NCCL_COMPRESS_ENABLE_THRESHOLD=10485760 # 10MB
export NCCL_COMPRESSORS_CONFIG_PATH=$COCCL_PATH/src/device/compress/configs_training
export NCCL_COMPRESSORS_LIB_PATH=$COCCL_PATH/build/obj/device/compress/libcompress
export NCCL_LOCAL_REGISTER=0
export NCCL_CHECKS_DISABLE=1
export NCCL_BUFFSIZE=33554432


NNODES=$1
GPUS_PER_NODE=$2
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=2
MICRO_BATCH_SIZE=8
# Calculate GLOBAL_BATCH_SIZE based on Accumulation Step=1

GLOBAL_BATCH_SIZE=256
VOCAB_FILE=$DATASET_PATH/vocab.json
MERGE_FILE=$DATASET_PATH/merges.txt
DATA_PATH=$DATASET_PATH/pile_text_document


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank 0 \
"

MODEL_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
"

TRAINING_ARGS="
    --tensor-model-parallel-size $TENSOR_PARALLEL_SIZE \
    --pipeline-model-parallel-size $PIPELINE_PARALLEL_SIZE \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters 20000 \
"

OPTIMIZER_ARGS="
    --lr 0.0003 \
    --lr-decay-iters 70000 \
    --lr-decay-style cosine \
    --min-lr 0.00003 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-08 \
    --weight-decay .1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --loss-scale 0 \
    --loss-scale-window 1000 \
    --hysteresis 2 \
    --min-loss-scale 1 \
    --bf16 \
    --use-distributed-optimizer \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --timing-log-level 2 \
    --save-interval 10002 \
    --eval-interval 100 \
    --eval-iters 10 \
    --log-timers-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --log-throughput \
"
QUANTIZE_ARGS="
    --no-async-tensor-model-parallel-allreduce \
    --recompute-activations \
    --recompute-granularity selective \
"

cd $Megatron_PATH

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $MODEL_ARGS \
    $TRAINING_ARGS \
    $OPTIMIZER_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $QUANTIZE_ARGS \
    --distributed-backend nccl 

