# CUDA_PATH=$1
# MPI_PATH=$2
# COCCL_PATH=$3
# NVCC_GENCODE=$4

source env.sh 
# $CUDA_PATH $MPI_PATH $COCCL_PATH
make -j src.build NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
# cp src/coccl-extend/compressor_plugin/zfp/build/lib/libzfp_compressor.so build/obj/coccl-extend/compressor_plugin/libcompress/
# cd tests/coccl-tests
# make -j MPI=1 MPI_HOME=$MPI_HOME CUDA_HOME=$NVHPC_CUDA_HOME NCCL_HOME=$NCCL_HOME NVCC_GENCODE=$NVCC_GENCODE
# "-gencode=arch=compute_90,code=compute_90"
