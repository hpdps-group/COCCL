#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH -N 4
#SBATCH --gres=gpu:8
#SBATCH -x g0016,g0019
# module load cuda/12.4

CUDA_PATH=$1
MPI_PATH=$2
COCCL_PATH=$3
# bash build.sh $CUDA_PATH $MPI_PATH $COCCL_PATH
COMPRESSORS=("sdp4bit" "tahquant")
export CUDA_HOME=$CUDA_PATH
export PATH=$CUDA_HOME/bin:$PATH
export CUDACXX=$CUDA_HOME/bin/nvcc
export NVHPC_CUDA_HOME=$CUDA_HOME
export LD_LIBRARY_PATH=$CUDA_HOME/lib64/:$LD_LIBRARY_PATH


export NCCL_HOME=$COCCL_PATH/build
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$NCCL_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$NCCL_HOME/include:$CPLUS_INCLUDE_PATH
export NCCL_COMPRESSORS_CONFIG_PATH=$COCCL_PATH/src/device/compress/configs
export NCCL_COMPRESSORS_LIB_PATH=$COCCL_PATH/build/obj/device/compress/libcompress

export MPI_HOME=$MPI_PATH
export PATH=$MPI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH
export MANPATH=$MPI_HOME/share/man:$MANPATH

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

cd $COCCL_PATH

for ((gpus=4; gpus<=8; gpus*=2));
do
echo '==========================================================alltoall tests=========================================================='
echo '=================================================================================================================================='
for ((i=0; i<1;i++))
do
echo "------------------------------------------------------alltoall native $gpus H800 GPUs------------------------------------------------------"
mpirun  -np $gpus \
        -x LD_LIBRARY_PATH=$CUDA_HOME/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_DEBUG_FILE=ncclcomp.%h \
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
        -x NCCL_PIPELINE_DEPTH=1 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       $COCCL_PATH/tests/coccl-tests/build/alltoall_p2p_perf -b 1MB -e 8G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done


for comp in "${COMPRESSORS[@]}"; do
  echo "================= Running tests with compressor: $comp ================="
  for ((pipe=1;pipe<=8;pipe=pipe*2))do
   for ((i=0; i<1; i++)); do
    echo "------------------------------------------------------alltoall comp $gpus H800 GPUs pipe $pipe [compressor=$comp]------------------------------------------------------"

    mpirun -np $gpus \
           -x LD_LIBRARY_PATH=$CUDA_HOME/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
           -x NCCL_DEBUG=WRAN \
           -x NCCL_DEBUG_FILE=ncclcomp.%h \
           -x NCCL_BUFFSIZE=16777216 \
           -x NCCL_ENABLE_COMPRESS=1 \
           -x NCCL_COMPRESSORS=$comp \
           -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
           -x NCCL_ALLTOALL_COMPRESSORS=$comp \
           -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
           -x NCCL_ALLREDUCE_COMPRESSORS=$comp \
           -x NCCL_ALLREDUCE_INTER_COMPRESSORS=$comp \
           -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
           -x NCCL_ALLGATHER_COMPRESSORS=$comp \
           -x NCCL_ALLGATHER_INTER_COMPRESSORS=$comp \
           -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
           -x NCCL_REDUCESCATTER_COMPRESSORS=$comp \
           -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=$comp \
           -x NCCL_LOCAL_REGISTER=1 \
           -x NCCL_PIPELINE_DEPTH=$pipe \
           -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
           -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
           $COCCL_PATH/tests/coccl-tests/build/alltoall_comp_overlap_perf -b 1MB -e 8G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
  done 
 done
done


echo '==========================================================allgather tests=========================================================='
echo '=================================================================================================================================='
echo '=================================================================================================================================='

for ((i=0; i<1;i++))
do
echo "------------------------------------------------------allgather native $gpus H800 GPUs------------------------------------------------------"
mpirun  -np $gpus \
        -x LD_LIBRARY_PATH=$CUDA_HOME/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_DEBUG_FILE=ncclcomp.%h \
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
        -x NCCL_PIPELINE_SIZE=1 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       $COCCL_PATH/tests/coccl-tests/build/all_gather_perf -b 1MB -e 8G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done

echo ' '
for comp in "${COMPRESSORS[@]}"; do
  echo "================= Running allgather with compressor: $comp ================="
for ((pipe=1;pipe<=8;pipe=pipe*2))do
  for ((i=0; i<1; i++)); do
    echo "------------------------------------------------------allgather comp $gpus H800 GPUs pipe $pipe [compressor=$comp]------------------------------------------------------"

    mpirun -np $gpus \
           -x LD_LIBRARY_PATH=$CUDA_HOME/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
           -x NCCL_DEBUG=WRAN \
           -x NCCL_DEBUG_FILE=ncclcomp.%h \
           -x NCCL_BUFFSIZE=16777216 \
           -x NCCL_ENABLE_COMPRESS=1 \
           -x NCCL_COMPRESSORS=$comp \
           -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
           -x NCCL_ALLTOALL_COMPRESSORS=$comp \
           -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
           -x NCCL_ALLREDUCE_COMPRESSORS=$comp \
           -x NCCL_ALLREDUCE_INTER_COMPRESSORS=$comp \
           -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
           -x NCCL_ALLGATHER_COMPRESSORS=$comp \
           -x NCCL_ALLGATHER_INTER_COMPRESSORS=$comp \
           -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
           -x NCCL_REDUCESCATTER_COMPRESSORS=$comp \
           -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=$comp \
           -x NCCL_LOCAL_REGISTER=1 \
           -x NCCL_PIPELINE_DEPTH=$pipe \
           -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
           -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
           $COCCL_PATH/tests/coccl-tests/build/all_gather_comp_overlap_perf -b 1MB -e 8G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
  done
done
done

echo '==========================================================reducescatter tests=========================================================='
echo '=================================================================================================================================='

echo ' '
for ((i=0; i<1;i++))
do
echo "------------------------------------------------------reducescatter native $gpus H800 GPUs------------------------------------------------------"
mpirun  -np $gpus \
        -x LD_LIBRARY_PATH=$CUDA_HOME/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_DEBUG_FILE=ncclcomp.%h \
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
        -x NCCL_PIPELINE_SIZE=1 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       $COCCL_PATH/tests/coccl-tests/build/reduce_scatter_perf -b 1MB -e 8G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done
echo ' '
for comp in "${COMPRESSORS[@]}"; do
echo "================= Running reduce_scatter with compressor: $comp ================="
for ((i=0; i<1;i++))
do
echo "------------------------------------------------------reducescatter comp ring $gpus H800 GPUs[compressor=$comp]------------------------------------------------------"
mpirun  -np $gpus \
        -x LD_LIBRARY_PATH=$CUDA_HOME/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_DEBUG_FILE=ncclcomp.%h \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=1 \
        -x NCCL_COMPRESSORS=$comp \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=$comp \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=$comp \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=$comp \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=$comp \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=$comp \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=$comp \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=$comp \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_SIZE=1 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
        $COCCL_PATH/tests/coccl-tests/build/reduce_scatter_comp_perf -b 1MB -e 8G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done

echo ' '

for ((pipe=1;pipe<=8;pipe=pipe*2))do
for ((i=0; i<1;i++))
do
echo "------------------------------------------------------reducescatter comp oneshot $gpus H800 GPUs pipe $pipe [compressor=$comp]------------------------------------------------------"
mpirun  -np $gpus \
        -x LD_LIBRARY_PATH=$CUDA_HOME/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_DEBUG_FILE=ncclcomp.%h \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=1 \
        -x NCCL_COMPRESSORS=$comp \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=$comp \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=$comp \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=$comp \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=$comp \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=$comp \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=$comp \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=$comp \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_DEPTH=$pipe \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
        $COCCL_PATH/tests/coccl-tests/build/reduce_scatter_comp_oneshot_overlap_perf -b 1MB -e 8G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done
done

echo ' '
for ((pipe=1;pipe<=8;pipe=pipe*2))do
for ((i=0; i<1;i++))
do
echo "------------------------------------------------------reducescatter comp twoshot new overlap  $gpus H800 GPUs pipe $pipe [compressor=$comp]------------------------------------------------------"
mpirun  -np $gpus \
        -x LD_LIBRARY_PATH=$CUDA_HOME/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_DEBUG_FILE=ncclcomp.%h \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=1 \
        -x NCCL_COMPRESSORS=$comp \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=$comp \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=$comp \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=$comp \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=$comp \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=$comp \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=$comp \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=$comp \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_DEPTH=$pipe \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
        $COCCL_PATH/tests/coccl-tests/build/reduce_scatter_comp_twoshot_tl_overlap_perf -b 1MB -e 8G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done
done
echo ' '

done

echo '==========================================================allreduce tests=========================================================='
echo '=================================================================================================================================='
echo '=================================================================================================================================='
for ((i=0; i<1;i++))
do
echo "------------------------------------------------------allreduce native $gpus H800 GPUs------------------------------------------------------"
mpirun  -np $gpus \
        -x LD_LIBRARY_PATH=$CUDA_HOME/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_DEBUG_FILE=ncclcomp.%h \
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
        -x NCCL_PIPELINE_SIZE=1 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       $COCCL_PATH/tests/coccl-tests/build/all_reduce_perf -b 4KB -e 8G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done
echo ' '

for comp in "${COMPRESSORS[@]}"; do
echo "================= Running allreduce with compressor: $comp ================="
for ((i=0; i<1;i++))
do
echo "------------------------------------------------------allreduce comp ring $gpus H800 GPUs[compressor=$comp]------------------------------------------------------"
mpirun  -np $gpus \
        -x LD_LIBRARY_PATH=$CUDA_HOME/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_DEBUG_FILE=ncclcomp.%h \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=1 \
        -x NCCL_COMPRESSORS=$comp \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=$comp \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=$comp \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=$comp \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=$comp \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=$comp \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=$comp \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=$comp \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_SIZE=1 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       $COCCL_PATH/tests/coccl-tests/build/all_reduce_comp_ring_perf -b 1MB -e 8G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done
echo ' '

for ((i=0; i<1;i++))
do
echo "------------------------------------------------------allreduce comp oneshot $gpus H800 GPUs[compressor=$comp]------------------------------------------------------"
mpirun  -np $gpus \
        -x LD_LIBRARY_PATH=$CUDA_HOME/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_DEBUG_FILE=ncclcomp.%h \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=1 \
        -x NCCL_COMPRESSORS=$comp \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=$comp \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=$comp \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=$comp \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=$comp \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=$comp \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=$comp \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=$comp \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_SIZE=1 \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       $COCCL_PATH/tests/coccl-tests/build/all_reduce_comp_oneshot_perf -b 4KB -e 32M -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done
echo ' '
for ((pipe=1;pipe<=8;pipe=pipe*2))do
for ((i=0; i<1;i++))
do
echo "------------------------------------------------------allreduce comp twoshot $gpus H800 GPUs pipe $pipe [compressor=$comp]------------------------------------------------------"
mpirun  -np $gpus \
        -x LD_LIBRARY_PATH=$CUDA_HOME/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_DEBUG_FILE=ncclcomp.%h \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=1 \
        -x NCCL_COMPRESSORS=$comp \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=$comp \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=$comp \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=$comp \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=$comp \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=$comp \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=$comp \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=$comp \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_DEPTH=$pipe \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       $COCCL_PATH/tests/coccl-tests/build/all_reduce_comp_twoshot_overlap_perf -b 1MB -e 8G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done
done
echo ' '
for ((pipe=1;pipe<=8;pipe=pipe*2))do
for ((i=0; i<1;i++))
do
echo "------------------------------------------------------allreduce comp tripleshot overlap $gpus H800 GPUs pipe $pipe [compressor=$comp]------------------------------------------------------"
mpirun  -np $gpus \
        -x LD_LIBRARY_PATH=$CUDA_HOME/lib64:$NCCL_HOME/lib:$MPI_HOME/lib \
        -x NCCL_DEBUG=WRAN \
        -x NCCL_DEBUG_FILE=ncclcomp.%h \
        -x NCCL_BUFFSIZE=16777216 \
        -x NCCL_ENABLE_COMPRESS=1 \
        -x NCCL_COMPRESSORS=$comp \
        -x NCCL_ENABLE_ALLTOALL_COMPRESS=1 \
        -x NCCL_ALLTOALL_COMPRESSORS=$comp \
        -x NCCL_ENABLE_ALLREDUCE_COMPRESS=1 \
        -x NCCL_ALLREDUCE_COMPRESSORS=$comp \
        -x NCCL_ALLREDUCE_INTER_COMPRESSORS=$comp \
        -x NCCL_ENABLE_ALLGATHER_COMPRESS=1 \
        -x NCCL_ALLGATHER_COMPRESSORS=$comp \
        -x NCCL_ALLGATHER_INTER_COMPRESSORS=$comp \
        -x NCCL_ENABLE_REDUCESCATTER_COMPRESS=1 \
        -x NCCL_REDUCESCATTER_COMPRESSORS=$comp \
        -x NCCL_REDUCESCATTER_INTER_COMPRESSORS=$comp \
        -x NCCL_LOCAL_REGISTER=1 \
        -x NCCL_PIPELINE_DEPTH=$pipe \
        -x NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH} \
        -x NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH} \
       $COCCL_PATH/tests/coccl-tests/build/all_reduce_comp_tripleshot_tl_overlap_perf -b 1MB -e 8G -f 2 -t 1 -g 1 -w 50 -n 100 -c 0
done
done
echo ' '

echo ' '
done
done
