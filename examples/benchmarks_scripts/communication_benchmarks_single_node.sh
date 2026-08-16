
if type module >/dev/null 2>&1; then
  module unload cuda
  module load cuda/12.4
fi
CUDA_PATH=/data/apps/cuda/12.4
COCCL_PATH=/data/home/scyb672/run/lxc/COCCL-migrate
NGPUS=4

# bash build.sh $CUDA_PATH $MPI_PATH $COCCL_PATH
COMPRESSORS=("sdp4bit" "cuzfp")
export CUDA_HOME=$CUDA_PATH
export PATH=$CUDA_HOME/bin:$PATH
export CUDACXX=$CUDA_HOME/bin/nvcc
export NVHPC_CUDA_HOME=$CUDA_HOME
export LD_LIBRARY_PATH=$CUDA_HOME/lib64/:$LD_LIBRARY_PATH


export NCCL_HOME=$COCCL_PATH/build
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$NCCL_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$NCCL_HOME/include:$CPLUS_INCLUDE_PATH
export NCCL_COMPRESSORS_CONFIG_PATH=$COCCL_PATH/examples/benchmarks_scripts/configs
export NCCL_COMPRESSORS_LIB_PATH=$COCCL_PATH/build/obj/device/compress/libcompress


# coccl environment variables
export NCCL_DEBUG=WARN
export NCCL_DEBUG_FILE=ncclcomp.%h
export NCCL_BUFFSIZE=16777216
export NCCL_ENABLE_COMPRESS=1
export NCCL_COMPRESSORS=sdp4bit,cuzfp
export NCCL_ENABLE_ALLTOALL_COMPRESS=1
export NCCL_ALLTOALL_COMPRESSORS=sdp4bit
export NCCL_ENABLE_ALLREDUCE_COMPRESS=1
export NCCL_ALLREDUCE_COMPRESSORS=sdp4bit
export NCCL_ALLREDUCE_INTER_COMPRESSORS=sdp4bit
export NCCL_ENABLE_ALLGATHER_COMPRESS=1
export NCCL_ALLGATHER_COMPRESSORS=sdp4bit
export NCCL_ALLGATHER_INTER_COMPRESSORS=sdp4bit
export NCCL_ENABLE_REDUCESCATTER_COMPRESS=1
export NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit
export NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit
export NCCL_LOCAL_REGISTER=1 
export NCCL_PIPELINE_DEPTH=1 
export NCCL_COMPRESSORS_CONFIG_PATH=${NCCL_COMPRESSORS_CONFIG_PATH}
export NCCL_COMPRESSORS_LIB_PATH=${NCCL_COMPRESSORS_LIB_PATH}

cd $COCCL_PATH

for ((gpus=4; gpus<=$NGPUS; gpus*=2));
do
echo '==========================================================alltoall tests=========================================================='
echo '=================================================================================================================================='

echo "------------------------------------------------------alltoall native $gpus A800 GPUs------------------------------------------------------"
  export NCCL_ENABLE_COMPRESS=0
  $COCCL_PATH/tests/coccl-tests/build/alltoall_p2p_perf -b 1MB -e 8G -f 2 -t $gpus -g 1 -w 10 -n 20 -c 0



for comp in "${COMPRESSORS[@]}"; do
  echo "================= Running tests with compressor: $comp ================="
  export NCCL_ENABLE_COMPRESS=1
  export NCCL_ALLTOALL_COMPRESSORS=$comp
  for ((pipe=1;pipe<=8;pipe=pipe*2))do
    echo "------------------------------------------------------alltoall comp $gpus A800 GPUs pipe $pipe [compressor=$comp]------------------------------------------------------"
    export NCCL_PIPELINE_DEPTH=$pipe
    $COCCL_PATH/tests/coccl-tests/build/alltoall_comp_overlap_perf -b 1MB -e 8G -f 2 -t $gpus -g 1 -w 10 -n 20 -c 0
 done
done


echo '==========================================================allgather tests=========================================================='
echo '=================================================================================================================================='

echo "------------------------------------------------------allgather native $gpus A800 GPUs------------------------------------------------------"
export NCCL_ENABLE_COMPRESS=0
$COCCL_PATH/tests/coccl-tests/build/all_gather_perf -b 1MB -e 8G -f 2 -t $gpus -g 1 -w 10 -n 20 -c 0

echo ' '
for comp in "${COMPRESSORS[@]}"; do
  echo "================= Running allgather with compressor: $comp ================="
  export NCCL_ENABLE_COMPRESS=1
  export NCCL_ALLGATHER_COMPRESSORS=$comp
  export NCCL_ALLGATHER_INTER_COMPRESSORS=$comp
  for ((pipe=1;pipe<=8;pipe=pipe*2))do
    echo "------------------------------------------------------allgather comp $gpus A800 GPUs pipe $pipe [compressor=$comp]------------------------------------------------------"
    export NCCL_PIPELINE_DEPTH=$pipe
    $COCCL_PATH/tests/coccl-tests/build/all_gather_comp_overlap_perf -b 1MB -e 8G -f 2 -t $gpus -g 1 -w 10 -n 20 -c 0
  done
done

echo '==========================================================reducescatter tests=========================================================='
echo '=================================================================================================================================='

echo ' '
echo "------------------------------------------------------reducescatter native $gpus A800 GPUs------------------------------------------------------"
export NCCL_ENABLE_COMPRESS=0
$COCCL_PATH/tests/coccl-tests/build/reduce_scatter_perf -b 1MB -e 8G -f 2 -t $gpus -g 1 -w 10 -n 20 -c 0

echo ' '
for comp in "${COMPRESSORS[@]}"; do
  echo "================= Running reduce_scatter with compressor: $comp ================="
  export NCCL_ENABLE_COMPRESS=1
  export NCCL_REDUCESCATTER_COMPRESSORS=$comp
  export NCCL_REDUCESCATTER_INTER_COMPRESSORS=$comp

  echo ' '
  for ((pipe=1;pipe<=8;pipe=pipe*2))do
    echo "------------------------------------------------------reducescatter comp oneshot $gpus A800 GPUs pipe $pipe [compressor=$comp]------------------------------------------------------"
    export NCCL_PIPELINE_DEPTH=$pipe
    $COCCL_PATH/tests/coccl-tests/build/reduce_scatter_comp_oneshot_perf -b 1MB -e 8G -f 2 -t $gpus -g 1 -w 10 -n 20 -c 0
  done

  echo ' '
done

echo '==========================================================allreduce tests=========================================================='
echo '=================================================================================================================================='

echo "------------------------------------------------------allreduce native $gpus A800 GPUs------------------------------------------------------"
export NCCL_ENABLE_COMPRESS=0
$COCCL_PATH/tests/coccl-tests/build/all_reduce_perf -b 4KB -e 8G -f 2 -t $gpus -g 1 -w 10 -n 20 -c 0

echo ' '
for comp in "${COMPRESSORS[@]}"; do
  echo "================= Running allreduce with compressor: $comp ================="
  export NCCL_ENABLE_COMPRESS=1
  export NCCL_ALLREDUCE_COMPRESSORS=$comp
  export NCCL_ALLREDUCE_INTER_COMPRESSORS=$comp
 
  echo ' '
  echo "------------------------------------------------------allreduce comp oneshot $gpus A800 GPUs[compressor=$comp]------------------------------------------------------"
  $COCCL_PATH/tests/coccl-tests/build/all_reduce_comp_oneshot_perf -b 4KB -e 32M -f 2 -t $gpus -g 1 -w 10 -n 20 -c 0

  echo ' '
  for ((pipe=1;pipe<=8;pipe=pipe*2))do
    echo "------------------------------------------------------allreduce comp twoshot $gpus A800 GPUs pipe $pipe [compressor=$comp]------------------------------------------------------"
    export NCCL_PIPELINE_DEPTH=$pipe
    $COCCL_PATH/tests/coccl-tests/build/all_reduce_comp_twoshot_overlap_perf -b 1MB -e 8G -f 2 -t $gpus -g 1 -w 10 -n 20 -c 0
  done

  echo ' '
done

done
