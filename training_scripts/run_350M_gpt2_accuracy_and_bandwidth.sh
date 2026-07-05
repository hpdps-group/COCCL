#!/bin/bash


GPUS=8
NODES=1

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
script_list=(
  "350M_accBaseline.sh"
  "350M_accCOCCL.sh"
)



for RANK_SCRIPT in "${script_list[@]}"
do
bash ${RANK_SCRIPT} ${NODES} ${GPUS}
wait
done