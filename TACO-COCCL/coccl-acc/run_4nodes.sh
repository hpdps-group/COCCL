#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH -N 2
#SBATCH --gres=gpu:4

module load gcc/12.2
module load cuda/12.2
module load openmpi/4.1.1

export CUDA_PATH=/data/apps/cuda/12.2
export PATH=$CUDA_PATH/bin:$PATH
export CUDACXX=$CUDA_PATH/bin/nvcc
export NVHPC_CUDA_HOME=/data/apps/cuda/12.2
export LD_LIBRARY_PATH=$CUDA_PATH/lib64/:$LD_LIBRARY_PATH

export OMPI_DIR=/data/apps/openmpi/4.1.1
export PATH=$OMPI_DIR/bin:$PATH
export MPI_HOME=/data/apps/openmpi/4.1.1
export LD_LIBRARY_PATH=$OMPI_DIR/lib:$LD_LIBRARY_PATH

export NCCL_HOME=/HOME/scw6doz/run/SC25_COCCL/coccl-new-overlap/build
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
export PATH=$NCCL_HOME/lib:$PATH


# export LD_LIBRARY_PATH=/HOME/scw6doz/run/lxc/nccl-comp-newbuff/build/obj/device/compress/libcompress:$LD_LIBRARY_PATH

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

GPUS=8
### Job ID
JOB_ID="${SLURM_JOB_ID}"

### hosfile
HOSTFILE_8GPU="hostfile_4nodes_8GPUeach.${JOB_ID}"

for i in `scontrol show hostnames`
do
  let k=k+1
  host[$k]=$i
  rank[$k]=$(($k-1))
  echo "${host[$k]} slots=8" >> $HOSTFILE_8GPU
done

export NCCL_DEBUG=WRAN

export NCCL_P2P_DISABLE=0
export NCCL_P2P_DIRECT_DISABLE=0
export NCCL_DEBUG_FILE=ncclcomp.%h
export NCCL_ENABLE_COMPRESS=1
export NCCL_COMPRESSORS=sdp4bit,minmaxUint8
export NCCL_ENABLE_ALLTOALL_COMPRESS=1
export NCCL_ALLTOALL_COMPRESSORS=sdp4bit
export NCCL_ENABLE_ALLREDUCE_COMPRESS=1
export NCCL_ALLREDUCE_COMPRESSORS=sdp4bit
export NCCL_ENABLE_ALLGATHER_COMPRESS=1
export NCCL_ALLGATHER_COMPRESSORS=sdp4bit
export NCCL_ENABLE_REDUCESCATTER_COMPRESS=1
export NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit
export NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit
export NCCL_COMPRESSORS_CONFIG_PATH=/HOME/scw6doz/run/SC25_COCCL/coccl-new-overlap/src/device/compress/configs
export NCCL_COMPRESSORS_LIB_PATH=/HOME/scw6doz/run/SC25_COCCL/coccl-new-overlap/build/obj/device/compress/libcompress

export NCCL_DEBUG=DEBUG
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3

export NCCL_CHECKS_DISABLE=1
export NCCL_LOCAL_REGISTER=1

cd /data/run01/scw6doz/SC25_COCCL/coccl-new-overlap

for ((ng=16; ng<=32; ng*=2)); 
do
echo '==========================================================alltoall tests=========================================================='
echo '=================================================================================================================================='
echo '=================================================================================================================================='
for ((i=0; i<3;i++))
do
echo "------------------------------------------------------alltoall native 4090-GPUs$ng ------------------------------------------------------"
mpirun  -hostfile "${HOSTFILE_8GPU}" \
        -np $ng \
        -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=${NCCL_IB_HCA} \
        -x NCCL_PXN_DISABLE=1 \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=0 \
        -x NCCL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=sdp4bit \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=sdp4bit \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_SIZE=0 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       /data/run01/scw6doz/SC25_COCCL/coccl-new-overlap-tests/build/alltoall_p2p_perf -b 256KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done
echo ' '
for ((i=0; i<3;i++))
do
echo "------------------------------------------------------alltoall native 4090-GPUs $ng ------------------------------------------------------"
mpirun  -hostfile "${HOSTFILE_8GPU}" \
        -np $ng \
         -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=${NCCL_IB_HCA} \
        -x NCCL_PXN_DISABLE=1 \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=1 \
        -x NCCL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=sdp4bit \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=sdp4bit \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_SIZE=0 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       /data/run01/scw6doz/SC25_COCCL/coccl-new-overlap-tests/build/alltoall_comp_perf -b 256KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done
echo '=================================================================================================================================='
echo '=================================================================================================================================='
echo '=================================================================================================================================='

echo ' '
echo ' '
echo ' '

echo '==========================================================allgather tests=========================================================='
echo '=================================================================================================================================='
echo '=================================================================================================================================='
for ((i=0; i<3;i++))
do
echo "------------------------------------------------------allgather native 4090-GPUs$ng ------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE_8GPU}" \
        -np $ng \
         -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=${NCCL_IB_HCA} \
        -x NCCL_PXN_DISABLE=1 \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=0 \
        -x NCCL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=sdp4bit \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=sdp4bit \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_SIZE=0 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       /data/run01/scw6doz/SC25_COCCL/coccl-new-overlap-tests/build/all_gather_perf -b 256KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done
echo ' '
for ((i=0; i<3;i++))
do
echo "------------------------------------------------------allgather comp 4090-GPUs$ng ------------------------------------------------------"
mpirun  -hostfile "${HOSTFILE_8GPU}" \
        -np $ng \
         -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=${NCCL_IB_HCA} \
        -x NCCL_PXN_DISABLE=1 \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=1 \
        -x NCCL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=sdp4bit \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=sdp4bit \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_SIZE=0 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       /data/run01/scw6doz/SC25_COCCL/coccl-new-overlap-tests/build/all_gather_comp_perf -b 256KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done

echo '=================================================================================================================================='
echo '=================================================================================================================================='
echo '=================================================================================================================================='

echo ' '
echo ' '
echo ' '

echo '==========================================================reducescatter tests====================================================='
echo '=================================================================================================================================='
echo '=================================================================================================================================='
for ((i=0; i<3;i++))
do
echo "------------------------------------------------------reducescatter native 4090 GPU$ng------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE_8GPU}" \
        -np $ng \
         -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=${NCCL_IB_HCA} \
        -x NCCL_PXN_DISABLE=1 \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=0 \
        -x NCCL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=sdp4bit \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=sdp4bit \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_SIZE=0 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       /data/run01/scw6doz/SC25_COCCL/coccl-new-overlap-tests/build/reduce_scatter_perf -b 256KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done
echo ' '
for ((i=0; i<3;i++))
do
echo "------------------------------------------------------reducescatter comp ring 4090 GPU$ng------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE_8GPU}" \
        -np $ng \
         -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=${NCCL_IB_HCA} \
        -x NCCL_PXN_DISABLE=1 \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=1 \
        -x NCCL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=sdp4bit \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=sdp4bit \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_SIZE=0 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       /data/run01/scw6doz/SC25_COCCL/coccl-new-overlap-tests/build/reduce_scatter_comp_perf -b 256KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done

for ((i=0; i<3;i++))
do

echo "------------------------------------------------------reducescatter comp oneshot 4090 GPU$ng------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE_8GPU}" \
        -np $ng \
         -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=${NCCL_IB_HCA} \
        -x NCCL_PXN_DISABLE=1 \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=1 \
        -x NCCL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=sdp4bit \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=sdp4bit \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_SIZE=0 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       /data/run01/scw6doz/SC25_COCCL/coccl-new-overlap-tests/build/reduce_scatter_comp_oneshot_perf -b 256KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done
echo ' '
for ((i=0; i<3;i++))
do
echo "------------------------------------------------------reducescatter comp twoshot 4090 GPU $ng------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE_8GPU}" \
        -np $ng \
         -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=${NCCL_IB_HCA} \
        -x NCCL_PXN_DISABLE=1 \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=1 \
        -x NCCL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=sdp4bit \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=sdp4bit \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_SIZE=0 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       /data/run01/scw6doz/SC25_COCCL/coccl-new-overlap-tests/build/reduce_scatter_comp_twoshot_perf -b 256KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done

for ((i=0; i<3;i++))
do
echo "------------------------------------------------------reducescatter comp twoshot new 4090 GPU $ng------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE_8GPU}" \
        -np $ng \
         -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=${NCCL_IB_HCA} \
        -x NCCL_PXN_DISABLE=1 \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=1 \
        -x NCCL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=sdp4bit \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=sdp4bit \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_SIZE=0 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       /data/run01/scw6doz/SC25_COCCL/coccl-new-overlap-tests/build/reduce_scatter_comp_twoshot_new_perf -b 256KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done

echo ' '

for ((i=0; i<3;i++))
do
echo "------------------------------------------------------reducescatter comp twoshot new 2pipe 4090 GPU $ng------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE_8GPU}" \
        -np $ng \
         -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=${NCCL_IB_HCA} \
        -x NCCL_PXN_DISABLE=1 \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=1 \
        -x NCCL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=sdp4bit \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=sdp4bit \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_SIZE=2 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       /data/run01/scw6doz/SC25_COCCL/coccl-new-overlap-tests/build/reduce_scatter_comp_twoshot_new_multi_perf -b 256KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done
echo ' '


for ((i=0; i<3;i++))
do
echo "------------------------------------------------------reducescatter comp twoshot new 4pipe 4090 GPU $ng------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE_8GPU}" \
        -np $ng \
         -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=${NCCL_IB_HCA} \
        -x NCCL_PXN_DISABLE=1 \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=1 \
        -x NCCL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=sdp4bit \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=sdp4bit \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_SIZE=4 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       /data/run01/scw6doz/SC25_COCCL/coccl-new-overlap-tests/build/reduce_scatter_comp_twoshot_new_multi_perf -b 256KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done
echo ' '



for ((i=0; i<3;i++))
do
echo "------------------------------------------------------reducescatter comp twoshot new 8pipe 4090 GPU $ng------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE_8GPU}" \
        -np $ng \
         -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=${NCCL_IB_HCA} \
        -x NCCL_PXN_DISABLE=1 \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=1 \
        -x NCCL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=sdp4bit \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=sdp4bit \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_SIZE=8 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       /data/run01/scw6doz/SC25_COCCL/coccl-new-overlap-tests/build/reduce_scatter_comp_twoshot_new_multi_perf -b 256KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done
echo ' '

for ((i=0; i<3;i++))
do

echo "------------------------------------------------------reduce scatter comp twoshot overlap p1 4090 GPU $ng------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE_8GPU}" \
        -np $ng \
         -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=${NCCL_IB_HCA} \
        -x NCCL_PXN_DISABLE=1 \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=1 \
        -x NCCL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=sdp4bit \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=sdp4bit \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_SIZE=4 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       /data/run01/scw6doz/SC25_COCCL/coccl-new-overlap-tests/build/reduce_scatter_comp_twoshot_multi_perf -b 256KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0

done
echo ' '

for ((i=0; i<3;i++))
do
echo "------------------------------------------------------reduce scatter comp twoshot overlap p2 4090 GPU $ng------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE_8GPU}" \
        -np $ng \
         -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=${NCCL_IB_HCA} \
        -x NCCL_PXN_DISABLE=1 \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=1 \
        -x NCCL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=sdp4bit \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=sdp4bit \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_SIZE=4 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       /data/run01/scw6doz/SC25_COCCL/coccl-new-overlap-tests/build/reduce_scatter_comp_twoshot_stage_perf -b 256KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done

echo '=================================================================================================================================='
echo '=================================================================================================================================='
echo '=================================================================================================================================='


echo ' '
echo ' '
echo ' '

echo '==========================================================allreduce tests====================================================='
echo '=================================================================================================================================='
echo '=================================================================================================================================='
for ((i=0; i<3;i++))
do
echo "------------------------------------------------------allreduce native 4090 GPU $ng------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE_8GPU}" \
        -np $ng \
         -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=${NCCL_IB_HCA} \
        -x NCCL_PXN_DISABLE=1 \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=0 \
        -x NCCL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=sdp4bit \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=sdp4bit \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_SIZE=0 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       /data/run01/scw6doz/SC25_COCCL/coccl-new-overlap-tests/build/all_reduce_perf -b 256KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done
echo ' '


for ((i=0; i<3;i++))
do

echo "------------------------------------------------------allreduce comp ring 4090 GPU $ng------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE_8GPU}" \
        -np $ng \
         -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=${NCCL_IB_HCA} \
        -x NCCL_PXN_DISABLE=1 \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=1 \
        -x NCCL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=sdp4bit \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=sdp4bit \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_SIZE=0 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       /data/run01/scw6doz/SC25_COCCL/coccl-new-overlap-tests/build/all_reduce_comp_ring_perf -b 256KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done

echo ' '
for ((i=0; i<3;i++))
do
echo "------------------------------------------------------allreduce comp oneshot 4090 GPU $ng------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE_8GPU}" \
        -np $ng \
         -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=${NCCL_IB_HCA} \
        -x NCCL_PXN_DISABLE=1 \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=1 \
        -x NCCL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=sdp4bit \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=sdp4bit \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_SIZE=0 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       /data/run01/scw6doz/SC25_COCCL/coccl-new-overlap-tests/build/all_reduce_comp_oneShot_perf -b 16K -e 16M -f 2 -t 1 -g 1 -w 50 -n 100 -c 0

done

echo ' '
for ((i=0; i<3;i++))
do
echo "------------------------------------------------------allreduce comp twoshot 4090 GPU $ng------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE_8GPU}" \
        -np $ng \
         -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=${NCCL_IB_HCA} \
        -x NCCL_PXN_DISABLE=1 \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=1 \
        -x NCCL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=sdp4bit \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=sdp4bit \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_SIZE=0 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       /data/run01/scw6doz/SC25_COCCL/coccl-new-overlap-tests/build/all_reduce_comp_twoShot_perf -b 256KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0

done


for ((i=0; i<3;i++))
do
echo "------------------------------------------------------allreduce comp tripleshot 4090 GPU $ng------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE_8GPU}" \
        -np $ng \
         -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=${NCCL_IB_HCA} \
        -x NCCL_PXN_DISABLE=1 \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=1 \
        -x NCCL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=sdp4bit \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=sdp4bit \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_SIZE=0 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       /data/run01/scw6doz/SC25_COCCL/coccl-new-overlap-tests/build/all_reduce_comp_tripleShot_perf -b 256KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0

done
echo ' '
for ((i=0; i<3;i++))
do
echo "------------------------------------------------------allreduce comp tripleshot new 2pipe 4090 GPU $ng------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE_8GPU}" \
        -np $ng \
         -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=${NCCL_IB_HCA} \
        -x NCCL_PXN_DISABLE=1 \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=1 \
        -x NCCL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=sdp4bit \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=sdp4bit \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_SIZE=2 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       /data/run01/scw6doz/SC25_COCCL/coccl-new-overlap-tests/build/all_reduce_comp_tripleShot_new_perf -b 256KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0

done
echo ' '
for ((i=0; i<3;i++))
do
echo "------------------------------------------------------allreduce comp tripleshot new 4pipe 4090 GPU $ng------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE_8GPU}" \
        -np $ng \
         -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=${NCCL_IB_HCA} \
        -x NCCL_PXN_DISABLE=1 \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=1 \
        -x NCCL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=sdp4bit \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=sdp4bit \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_SIZE=4 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       /data/run01/scw6doz/SC25_COCCL/coccl-new-overlap-tests/build/all_reduce_comp_tripleShot_new_perf -b 256KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0

done
echo ' '
for ((i=0; i<3;i++))
do
echo "------------------------------------------------------allreduce comp tripleshot new 8pipe 4090 GPU $ng------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE_8GPU}" \
        -np $ng \
         -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=${NCCL_IB_HCA} \
        -x NCCL_PXN_DISABLE=1 \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=1 \
        -x NCCL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=sdp4bit \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=sdp4bit \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_SIZE=8 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       /data/run01/scw6doz/SC25_COCCL/coccl-new-overlap-tests/build/all_reduce_comp_tripleShot_new_perf -b 256KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0

done



echo '=================================================================================================================================='
echo '=================================================================================================================================='
echo '=================================================================================================================================='


done