#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH -N 2
#SBATCH --gres=gpu:4


module load compilers/cuda/11.8
module load compilers/gcc/11.3.0
module load ucx/1.12.1_cuda11.8
module load mpi/openmpi4.1.5-gcc11.3.0-cuda11.8-ucx1.12.1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

GPUS=4
### Job ID
JOB_ID="${SLURM_JOB_ID}"

### hosfile
HOSTFILE="hostfile_16nodes.${JOB_ID}"

for i in `scontrol show hostnames`
do
  let k=k+1
  host[$k]=$i
  rank[$k]=$(($k-1))
  echo "${host[$k]} slots=4" >> $HOSTFILE
done

export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_DEBUG=INFO
export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_ALGO=Ring


cd /home/bingxing2/home/scx9kvs/SC25_COCCL/coccl


export CUDA_PATH=/home/bingxing2/apps/compilers/cuda/cuda-11.8
export PATH=$CUDA_PATH/bin:$PATH
export CUDACXX=$CUDA_PATH/bin/nvcc
# export NVHPC_CUDA_HOME=/home/bingxing2/apps/compilers/cuda/cuda-11.8
export LD_LIBRARY_PATH=$CUDA_PATH/lib64/:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$CUDA_PATH/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDA_PATH/include:$CPLUS_INCLUDE_PATH

export NCCL_HOME=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/build
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$NCCL_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$NCCL_HOME/include:$CPLUS_INCLUDE_PATH

export MPI_HOME=/home/bingxing2/apps/mpi/4.1.5-gcc11.3.0-cuda11.8-ucx1.12.1

for ((ng=32; ng<=64; ng*=2)); 
do
echo "==========================================================alltoall tests A100-GPU$ng=========================================================="
echo "=================================================================================================================================="
echo "=================================================================================================================================="

for ((i=0; i<5;i++))
do
echo "------------------------------------------------------alltoall native A100-GPUs$ng ------------------------------------------------------"
mpirun  -hostfile "${HOSTFILE}" \
        -np $ng \
        -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib \
        -x NCCL_DEBUG=DEBUG \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=mlx5 \
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
        -x NCCL_COMPRESSORS_CONFIG_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/src/device/compress/configs \
        -x NCCL_COMPRESSORS_LIB_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/build/obj/device/compress/libcompress \
        /home/bingxing2/home/scx9kvs/SC25_COCCL/coccl-tests/build/alltoall_perf -b 512KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done
echo " "
echo " "
echo " "

for ((i=0; i<5;i++))
do
echo "------------------------------------------------------alltoall comp A100-GPUs$ng ------------------------------------------------------"
mpirun  -hostfile "${HOSTFILE}" \
        -np $ng \
        -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib \
        -x NCCL_DEBUG=DEBUG \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=mlx5 \
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
        -x NCCL_COMPRESSORS_CONFIG_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/src/device/compress/configs \
        -x NCCL_COMPRESSORS_LIB_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/build/obj/device/compress/libcompress \
        /home/bingxing2/home/scx9kvs/SC25_COCCL/coccl-tests/build/alltoall_comp_perf -b 512KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done
echo " "
echo "=================================================================================================================================="
echo "=================================================================================================================================="
echo "=================================================================================================================================="


echo " "
echo " "
echo " "

echo " "
echo " "
echo " "

echo " "
echo " "
echo " "

echo "==========================================================allgather tests A100- $ng GPU=========================================================="
echo "=================================================================================================================================="
echo "=================================================================================================================================="

echo "------------------------------------------------------allgather native A100- $ng GPU------------------------------------------------------"
for ((i=0; i<5;i++))
do
mpirun  -hostfile "${HOSTFILE}" \
        -np $ng \
        -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib \
        -x NCCL_DEBUG=DEBUG \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=mlx5 \
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
        -x NCCL_COMPRESSORS_CONFIG_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/src/device/compress/configs \
        -x NCCL_COMPRESSORS_LIB_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/build/obj/device/compress/libcompress \
        /home/bingxing2/home/scx9kvs/SC25_COCCL/coccl-tests/build/all_gather_perf -b 512KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done

for ((i=0; i<5;i++))
do
echo "------------------------------------------------------allgather comp A100- $ng GPU------------------------------------------------------"
mpirun  -hostfile "${HOSTFILE}" \
        -np $ng \
        -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib \
        -x NCCL_DEBUG=DEBUG \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=mlx5 \
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
        -x NCCL_COMPRESSORS_CONFIG_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/src/device/compress/configs \
        -x NCCL_COMPRESSORS_LIB_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/build/obj/device/compress/libcompress \
        /home/bingxing2/home/scx9kvs/SC25_COCCL/coccl-tests/build/all_gather_comp_perf -b 512KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done
echo "=================================================================================================================================="
echo "=================================================================================================================================="
echo "=================================================================================================================================="

echo " "
echo " "
echo " "


echo " "
echo " "
echo " "


echo " "
echo " "
echo " "

echo "==========================================================reducescatter tests A100- $ng GPU====================================================="
echo "=================================================================================================================================="
echo "=================================================================================================================================="

for ((i=0; i<5;i++))
do
echo "------------------------------------------------------reducescatter native A100- $ng GPU------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE}" \
        -np $ng \
        -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib \
        -x NCCL_DEBUG=DEBUG \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=mlx5 \
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
        -x NCCL_COMPRESSORS_CONFIG_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/src/device/compress/configs \
        -x NCCL_COMPRESSORS_LIB_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/build/obj/device/compress/libcompress \
        /home/bingxing2/home/scx9kvs/SC25_COCCL/coccl-tests/build/reduce_scatter_perf -b 512KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done

echo " "
echo " "
echo " "


for ((i=0; i<5;i++))
do
echo "------------------------------------------------------reducescatter comp ring A100- $ng GPU------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE}" \
        -np $ng \
        -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib \
        -x NCCL_DEBUG=DEBUG \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=mlx5 \
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
        -x NCCL_COMPRESSORS_CONFIG_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/src/device/compress/configs \
        -x NCCL_COMPRESSORS_LIB_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/build/obj/device/compress/libcompress \
        /home/bingxing2/home/scx9kvs/SC25_COCCL/coccl-tests/build/reduce_scatter_comp_perf -b 512KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done
echo " "
echo " "
echo " "


for ((i=0; i<5;i++))
do
echo "------------------------------------------------------reducescatter comp oneshot A100- $ng GPU------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE}" \
        -np $ng \
        -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib \
        -x NCCL_DEBUG=DEBUG \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=mlx5 \
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
        -x NCCL_COMPRESSORS_CONFIG_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/src/device/compress/configs \
        -x NCCL_COMPRESSORS_LIB_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/build/obj/device/compress/libcompress \
        /home/bingxing2/home/scx9kvs/SC25_COCCL/coccl-tests/build/reduce_scatter_comp_oneshot_perf -b 512KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done

echo " "
echo " "
echo " "

for ((i=0; i<5;i++))
do
echo "------------------------------------------------------reducescatter comp twoshot A100- $ng GPU------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE}" \
        -np $ng \
        -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib \
        -x NCCL_DEBUG=DEBUG \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=mlx5 \
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
        -x NCCL_COMPRESSORS_CONFIG_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/src/device/compress/configs \
        -x NCCL_COMPRESSORS_LIB_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/build/obj/device/compress/libcompress \
        /home/bingxing2/home/scx9kvs/SC25_COCCL/coccl-tests/build/reduce_scatter_comp_twoshot_perf -b 512KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done

echo " "
echo " "
echo " "

for ((i=0; i<5;i++))
do
echo "------------------------------------------------------reducescatter comp twoshot new A100- $ng GPU------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE}" \
        -np $ng \
        -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib \
        -x NCCL_DEBUG=DEBUG \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=mlx5 \
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
        -x NCCL_COMPRESSORS_CONFIG_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/src/device/compress/configs \
        -x NCCL_COMPRESSORS_LIB_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/build/obj/device/compress/libcompress \
        /home/bingxing2/home/scx9kvs/SC25_COCCL/coccl-tests/build/reduce_scatter_comp_twoshot_new_perf -b 512KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done

echo " "
echo " "
echo " "

for ((i=0; i<5;i++))
do
echo "------------------------------------------------------reducescatter comp twoshot new overlap 2pipe A100- $ng GPU------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE}" \
        -np $ng \
        -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib \
        -x NCCL_DEBUG=DEBUG \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=mlx5 \
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
        -x NCCL_COMPRESSORS_CONFIG_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/src/device/compress/configs \
        -x NCCL_COMPRESSORS_LIB_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/build/obj/device/compress/libcompress \
        /home/bingxing2/home/scx9kvs/SC25_COCCL/coccl-tests/build/reduce_scatter_comp_twoshot_new_multi_perf -b 512KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done

echo " "
echo " "
echo " "

for ((i=0; i<5;i++))
do
echo "------------------------------------------------------reducescatter comp twoshot new overlap 4pipe A100- $ng GPU------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE}" \
        -np $ng \
        -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib \
        -x NCCL_DEBUG=DEBUG \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=mlx5 \
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
        -x NCCL_COMPRESSORS_CONFIG_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/src/device/compress/configs \
        -x NCCL_COMPRESSORS_LIB_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/build/obj/device/compress/libcompress \
        /home/bingxing2/home/scx9kvs/SC25_COCCL/coccl-tests/build/reduce_scatter_comp_twoshot_new_multi_perf -b 512KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done

echo " "
echo " "
echo " "

for ((i=0; i<5;i++))
do
echo "------------------------------------------------------reducescatter comp twoshot new overlap 8pipe A100- $ng GPU------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE}" \
        -np $ng \
        -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib \
        -x NCCL_DEBUG=DEBUG \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=mlx5 \
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
        -x NCCL_COMPRESSORS_CONFIG_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/src/device/compress/configs \
        -x NCCL_COMPRESSORS_LIB_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/build/obj/device/compress/libcompress \
        /home/bingxing2/home/scx9kvs/SC25_COCCL/coccl-tests/build/reduce_scatter_comp_twoshot_new_multi_perf -b 512KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done

echo " "
echo " "
echo " "

for ((i=0; i<5;i++))
do
echo "------------------------------------------------------reduce scatter comp twoshot overlap p1 A100- $ng GPU------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE}" \
        -np $ng \
        -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib \
        -x NCCL_DEBUG=DEBUG \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=mlx5 \
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
        -x NCCL_COMPRESSORS_CONFIG_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/src/device/compress/configs \
        -x NCCL_COMPRESSORS_LIB_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/build/obj/device/compress/libcompress \
        /home/bingxing2/home/scx9kvs/SC25_COCCL/coccl-tests/build/reduce_scatter_comp_twoshot_multi_perf -b 512KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done

echo " "
echo " "
echo " "

for ((i=0; i<5;i++))
do
echo "------------------------------------------------------reduce scatter comp twoshot overlap p2 A100- $ng GPU------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE}" \
        -np $ng \
        -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib \
        -x NCCL_DEBUG=DEBUG \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=mlx5 \
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
        -x NCCL_COMPRESSORS_CONFIG_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/src/device/compress/configs \
        -x NCCL_COMPRESSORS_LIB_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/build/obj/device/compress/libcompress \
        /home/bingxing2/home/scx9kvs/SC25_COCCL/coccl-tests/build/reduce_scatter_comp_twoshot_stage_perf -b 512KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done

echo "=================================================================================================================================="
echo "=================================================================================================================================="
echo "=================================================================================================================================="


echo " "
echo " "
echo " "

echo " "
echo " "
echo " "

echo " "
echo " "
echo " "

echo "==========================================================allreduce tests A100- $ng GPU====================================================="
echo "=================================================================================================================================="
echo "=================================================================================================================================="

echo "------------------------------------------------------allreduce native A100- $ng GPU------------------------------------------------------"

for ((i=0; i<5;i++))
do
echo "------------------------------------------------------allreduce native A100- $ng GPU------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE}" \
        -np $ng \
        -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib \
        -x NCCL_DEBUG=DEBUG \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=mlx5 \
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
        -x NCCL_COMPRESSORS_CONFIG_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/src/device/compress/configs \
        -x NCCL_COMPRESSORS_LIB_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/build/obj/device/compress/libcompress \
        /home/bingxing2/home/scx9kvs/SC25_COCCL/coccl-tests/build/all_reduce_perf -b 512KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done
echo " "
echo " "
echo " "


for ((i=0; i<5;i++))
do
echo "------------------------------------------------------allreduce comp ring A100- $ng GPU------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE}" \
        -np $ng \
        -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib \
        -x NCCL_DEBUG=DEBUG \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=mlx5 \
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
        -x NCCL_COMPRESSORS_CONFIG_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/src/device/compress/configs \
        -x NCCL_COMPRESSORS_LIB_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/build/obj/device/compress/libcompress \
        /home/bingxing2/home/scx9kvs/SC25_COCCL/coccl-tests/build/all_reduce_comp_ring_perf -b 512KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done
echo " "
echo " "
echo " "

for ((i=0; i<5;i++))
do
echo "------------------------------------------------------allreduce comp oneshot A100- $ng GPU------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE}" \
        -np $ng \
        -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib \
        -x NCCL_DEBUG=DEBUG \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=mlx5 \
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
        -x NCCL_COMPRESSORS_CONFIG_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/src/device/compress/configs \
        -x NCCL_COMPRESSORS_LIB_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/build/obj/device/compress/libcompress \
        /home/bingxing2/home/scx9kvs/SC25_COCCL/coccl-tests/build/all_reduce_comp_oneShot_perf -b 16K -e 16M -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done

echo " "
echo " "
echo " "


for ((i=0; i<5;i++))
do
echo "------------------------------------------------------allreduce comp twoshot A100- $ng GPU------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE}" \
        -np $ng \
        -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib \
        -x NCCL_DEBUG=DEBUG \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=mlx5 \
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
        -x NCCL_COMPRESSORS_CONFIG_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/src/device/compress/configs \
        -x NCCL_COMPRESSORS_LIB_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/build/obj/device/compress/libcompress \
        /home/bingxing2/home/scx9kvs/SC25_COCCL/coccl-tests/build/all_reduce_comp_twoShot_perf -b 512KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done

echo " "
echo " "
echo " "

for ((i=0; i<5;i++))
do
echo "------------------------------------------------------all reduce comp tripleshot overlap A100- $ng GPU------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE}" \
        -np $ng \
        -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib \
        -x NCCL_DEBUG=DEBUG \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=mlx5 \
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
        -x NCCL_COMPRESSORS_CONFIG_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/src/device/compress/configs \
        -x NCCL_COMPRESSORS_LIB_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/build/obj/device/compress/libcompress \
        /home/bingxing2/home/scx9kvs/SC25_COCCL/coccl-tests/build/all_reduce_comp_tripleShot_perf -b 512KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done


echo " "
echo " "
echo " "

for ((i=0; i<5;i++))
do
echo "------------------------------------------------------all reduce comp tripleshot overlap new 2pipe A100- $ng GPU------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE}" \
        -np $ng \
        -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib \
        -x NCCL_DEBUG=DEBUG \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=mlx5 \
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
        -x NCCL_COMPRESSORS_CONFIG_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/src/device/compress/configs \
        -x NCCL_COMPRESSORS_LIB_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/build/obj/device/compress/libcompress \
        /home/bingxing2/home/scx9kvs/SC25_COCCL/coccl-tests/build/all_reduce_comp_tripleShot_new_perf -b 512KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done

echo " "
echo " "
echo " "

for ((i=0; i<5;i++))
do
echo "------------------------------------------------------all reduce comp tripleshot overlap new 4pipe A100- $ng GPU------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE}" \
        -np $ng \
        -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib \
        -x NCCL_DEBUG=DEBUG \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=mlx5 \
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
        -x NCCL_COMPRESSORS_CONFIG_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/src/device/compress/configs \
        -x NCCL_COMPRESSORS_LIB_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/build/obj/device/compress/libcompress \
        /home/bingxing2/home/scx9kvs/SC25_COCCL/coccl-tests/build/all_reduce_comp_tripleShot_new_perf -b 512KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done

echo " "
echo " "
echo " "

for ((i=0; i<5;i++))
do
echo "------------------------------------------------------all reduce comp tripleshot overlap new 8pipe A100- $ng GPU------------------------------------------------------"

mpirun  -hostfile "${HOSTFILE}" \
        -np $ng \
        -x LD_LIBRARY_PATH=$CUDA_PATH/lib64:$NCCL_HOME/lib \
        -x NCCL_DEBUG=DEBUG \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_TIMEOUT=23 \
        -x NCCL_IB_RETRY_CNT=13 \
        -x NCCL_IB_HCA=mlx5 \
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
        -x NCCL_COMPRESSORS_CONFIG_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/src/device/compress/configs \
        -x NCCL_COMPRESSORS_LIB_PATH=/home/bingxing2/home/scx9kvs/SC25_COCCL/coccl/build/obj/device/compress/libcompress \
        /home/bingxing2/home/scx9kvs/SC25_COCCL/coccl-tests/build/all_reduce_comp_tripleShot_new_perf -b 512KB -e 4G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done

echo "=================================================================================================================================="
echo "=================================================================================================================================="
echo "=================================================================================================================================="

echo " "
echo " "
echo " "

done

