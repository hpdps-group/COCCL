source env.sh
make -j src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
cp src/device/compress/zfp/build/lib/libzfp_compressor.so build/obj/device/compress/libcompress/
cd tests/coccl-tests
make -j MPI=1 MPI_HOME=$MPI_HOME CUDA_HOME=$NVHPC_CUDA_HOME NCCL_HOME=$NCCL_HOME NVCC_GENCODE="-gencode=arch=compute_90,code=compute_90"
#ASAN=1
