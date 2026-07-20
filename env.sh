module load cuda/12.8
module load openmpi/4.1.5_cuda12.8

# CUDA_PATH=$1
# MPI_PATH=$2
# COCCL_PATH=$3

# echo "$CUDA_PATH,$MPI_PATH,$COCCL_PATH"

export CUDA_HOME=/data/apps/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export CUDACXX=$CUDA_HOME/bin/nvcc
export NVHPC_CUDA_HOME=$CUDA_HOME
export LD_LIBRARY_PATH=$CUDA_HOME/lib64/:$LD_LIBRARY_PATH

export NCCL_HOME=/data/home/scyb672/run/lxc/coccl-refine/build
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$NCCL_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$NCCL_HOME/include:$CPLUS_INCLUDE_PATH

export MPI_HOME=/data/apps/openmpi/4.1.5_cuda12.8
export PATH=$MPI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH
export MANPATH=$MPI_HOME/share/man:$MANPATH

