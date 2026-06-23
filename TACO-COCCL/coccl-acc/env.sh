module load cuda/12.4

export CUDA_PATH=/data/apps/cuda/12.4
export PATH=$CUDA_PATH/bin:$PATH
export CUDACXX=$CUDA_PATH/bin/nvcc
export NVHPC_CUDA_HOME=/data/apps/cuda/12.4
export LD_LIBRARY_PATH=$CUDA_PATH/lib64/:$LD_LIBRARY_PATH

export NCCL_HOME=/data/home/scyb226/lxc/coccl-comm-speed/build
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$NCCL_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$NCCL_HOME/include:$CPLUS_INCLUDE_PATH

export MPI_HOME=/data/home/scyb226/lxc/openmpi
export PATH=$MPI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH
export MANPATH=$MPI_HOME/share/man:$MANPATH

