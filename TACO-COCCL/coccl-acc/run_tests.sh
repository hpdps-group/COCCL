source env.sh
bash build.sh
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# export NCCL_P2P_DISABLE=1
# export NCCL_SHM_DISABLE=1
# export NCCL_P2P_DIRECT_DISABLE=1
# export NCCL_NCHANNELS_PER_NET_PEER
# export NCCL_MIN_CTAS=32
# export NCCL_COMM_BLOCKING=0
# export NCCL_IB_DISABLE=0
# export NCCL_IB_HCA=mlx5_bond_0
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_GID_INDEX=3
# export NCCL_TOPO_DUMP_FILE=topo.xml
# compute-sanitizer --tool memcheck --log-file memcheck.log
# compute-sanitizer --tool memcheck --log-file memcheck.log 
# export NCCL_P2P_DISABLE=1
# export NCCL_SHM_DISABLE=1
# export NCCL_P2P_DIRECT_DISABLE=1
export NCCL_DEBUG_FILE=ncclcomp.%h
export NCCL_GRAPH_REGISTER=1
export NCCL_BUFFSIZE=33554432
export NCCL_LOCAL_REGISTER=1
export NCCL_ENABLE_COMPRESS=1
export NCCL_COMPRESSORS=sdp4bit,tahquant,zfp_compressor,fp8scale
export NCCL_ENABLE_ALLTOALL_COMPRESS=1
export NCCL_ALLTOALL_COMPRESSORS=sdp4bit 
export NCCL_ENABLE_ALLGATHER_COMPRESS=1
export NCCL_ALLGATHER_COMPRESSORS=sdp4bit 
export NCCL_ENABLE_REDUCESCATTER_COMPRESS=1
export NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit
export NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit
export NCCL_ENABLE_ALLREDUCE_COMPRESS=1
export NCCL_ALLREDUCE_COMPRESSORS=sdp4bit
export NCCL_ALLREDUCE_INTER_COMPRESSORS=sdp4bit
export NCCL_ENABLE_SENDRECV_COMPRESS=1
export NCCL_SENDRECV_COMPRESSORS=tahquant
export NCCL_SENDRECV_BWD_COMPRESSORS=tahquant
export NCCL_PIPELINE_DEPTH=1
export NCCL_PIPELINE_LEVEL=1
export NCCL_ENABLE_CHECK=1
# export NCCL_COMM_BLOCKING=1
export NCCL_COMPRESSORS_CONFIG_PATH=/data/home/scyb226/lm/coccl-acc/src/device/compress/configs
export NCCL_COMPRESSORS_LIB_PATH=/data/home/scyb226/lm/coccl-comm-speed/build/obj/device/compress/libcompress
# echo '------------------------------------------------------baseline perf------------------------------------------------------'
# ./tests/coccl-tests/build/sendrecv_perf -b 939524096 -e 4G -f 2 -t 2 -g 1 -w 20 -n 100 -c 1

# echo '------------------------------------------------------fp8scale sendrecv perf------------------------------------------------------'
#  ./tests/coccl-tests/build/sendrecv_comp_perf -b 16K -e 64K -f 2 -t 8 -g 1 -w 0 -n 1 -c 1

echo '------------------------------------------------------tahquant perf------------------------------------------------------'
./tests/coccl-tests/build/sendrecv_comp_perf -b 16K -e 64K -d bfloat16 -f 2 -t 2 -g 1 -w 20 -n 100 -c 1


# echo '------------------------------------------------------native alltoall perf------------------------------------------------------'
# ./tests/coccl-tests/build/alltoall_perf -b 4M -e 4G -f 2 -t 8 -g 1 -w 20 -n 100 -c 1


# echo '------------------------------------------------------sdp4bit alltoall perf------------------------------------------------------'
# ./tests/coccl-tests/build/alltoall_comp_perf -b 64K -e 64K -f 2 -t 8 -g 1 -w 0 -n 1 -c 1

# echo '------------------------------------------------------fp8scale alltoall perf------------------------------------------------------'
# ./tests/coccl-tests/build/alltoall_comp_perf -b 64K -e 64K -f 2 -t 8 -g 1 -w 0 -n 1 -c 1

# echo '------------------------------------------------------fp8scale allreduce perf------------------------------------------------------'
#  ./tests/coccl-tests/build/all_reduce_comp_twoshot_perf -b 64K -e 64K -f 2 -t 8 -g 8 -w 0 -n 1 -c 1

# echo '------------------------------------------------------fp8scale allgather perf------------------------------------------------------'
# ./tests/coccl-tests/build/all_gather_comp_perf -b 64K -e 64K -f 2 -t 8 -g 1 -w 0 -n 1 -c 

# echo '------------------------------------------------------fp8scale reducescatter perf------------------------------------------------------'
# ./tests/coccl-tests/build/reduce_scatter_comp_oneshot_perf -b 64K -e 64K -f 2 -t 8 -g 1 -w 0 -n 1 -c 1

# echo '------------------------------------------------------naive allgather perf------------------------------------------------------'
# ./tests/coccl-tests/build/all_gather_perf  -b 8M -e 8G -f 2 -t 8 -g 1 -w 20 -n 100 -c 0 -G 1


# echo '------------------------------------------------------sdp4bit allgather perf------------------------------------------------------'
# ./tests/coccl-tests/build/all_gather_comp_perf -b 64K -e 64K -f 2 -t 8 -g 1 -w 0 -n 1 -c 1

# echo '------------------------------------------------------fp8quan allgather perf------------------------------------------------------'
# ./tests/coccl-tests/build/all_gather_comp_perf -b 64K -e 64K -f 2 -t 8 -g 1 -w 0 -n 1 -c 1


# echo '------------------------------------------------------sdp4bit allgather overlap perf------------------------------------------------------'
# # nsys profile --force-overwrite true -o allgather_comp_overlap_8G 
# ./tests/coccl-tests/build/all_gather_comp_overlap_perf -b 1M -e 8G -f 2 -t 8 -g 1 -w 20 -n 100 -c 0

# ./tests/coccl-tests/build/reduce_scatter_comp_perf -d float -b 1M -e 1M -f 2 -t 8 -g 1 -w 0 -n 1 -c 1

# echo '------------------------------------------------------zfp reducescatter perf------------------------------------------------------'
# ./tests/coccl-tests/build/reduce_scatter_comp_twoshot_new_perf -d bfloat16 -b 256K -e 256K -f 2 -t 8 -g 1 -w 0 -n 1 -c 1


# echo '------------------------------------------------------naive reducescatter perf------------------------------------------------------'
# ./tests/coccl-tests/build/reduce_scatter_perf -b 4M -e 8G -f 2 -t 8 -g 1 -w 20 -n 100 -c 0

# echo '------------------------------------------------------sdp4bit reducescatter perf------------------------------------------------------'
# ./tests/coccl-tests/build/reduce_scatter_comp_oneshot_perf -b 64K -e 64K -f 2 -t 8 -g 1 -w 0 -n 1 -c 1

# echo '------------------------------------------------------fp8quan reducescatter perf------------------------------------------------------'
# ./tests/coccl-tests/build/reduce_scatter_comp_oneshot_perf -b 64K -e 64K -f 2 -t 8 -g 1 -w 0 -n 1 -c 1

# echo '------------------------------------------------------sdp4bit reducescatter overlap perf------------------------------------------------------'
# # nsys profile --force-overwrite true -o rs_oneshot_comp_overlap_8G  
# ./tests/coccl-tests/build/reduce_scatter_comp_oneshot_overlap_perf -b 4M -e 8G -f 2 -t 8 -g 1 -w 20 -n 100 -c 0

# echo '------------------------------------------------------sdp4bit reducescatter twoshot overlap perf------------------------------------------------------'
# # nsys profile --force-overwrite true -o rs_twoshot_comp_overlap_8G 
# ./tests/coccl-tests/build/reduce_scatter_comp_twoshot_tl_overlap_perf -b 4M -e 8G -f 2 -t 8 -g 1 -w 20 -n 100 -c 0

# echo '------------------------------------------------------naive allreduce perf------------------------------------------------------'
# ./tests/coccl-tests/build/all_reduce_perf -b 4M -e 4G -f 2 -t 4 -g 1 -w 20 -n 100 -c 0

# echo '------------------------------------------------------sdp4bit allreduce perf------------------------------------------------------'
#  ./tests/coccl-tests/build/all_reduce_comp_twoshot_perf -b 64K -e 64K -f 2 -t 8 -g 1 -w 0 -n 1 -c 1

# echo '------------------------------------------------------sdp4bit allreduce perf------------------------------------------------------'
# nsys profile --force-overwrite true -o ar_twoshot_comp_overlap_8G ./tests/coccl-tests/build/all_reduce_comp_twoshot_overlap_perf -b 16M -e 16M -f 2 -t 8 -g 1 -w 20 -n 100 -c 0

# echo '------------------------------------------------------sdp4bit allreduce triple perf------------------------------------------------------'
# # nsys profile --force-overwrite true -o ar_tripleshot_comp_overlap_8G 
# ./tests/coccl-tests/build/all_reduce_comp_tripleshot_perf -b 8G -e 8G -f 2 -t 8 -g 1 -w 20 -n 100 -c 0

#echo '------------------------------------------------------sdp4bit allreduce triple overlap perf------------------------------------------------------'
#nsys profile --force-overwrite true -o ar_tripleshot_comp_tl_overlap_8G ./tests/coccl-tests/build/all_reduce_comp_tripleshot_tl_overlap_perf -b 8G -e 8G -f 2 -t 8 -g 1 -w 20 -n 100 -c 0

# echo '------------------------------------------------------native alltoall perf------------------------------------------------------'
# ./tests/coccl-tests/build/alltoall_perf -b 8M -e 8M -f 2 -t 8 -g 1 -w 20 -n 100 -c 0 

# echo '------------------------------------------------------alltoall comp perf------------------------------------------------------'
# # # nsys profile --force-overwrite true -o alltoall_comp_nonoverlap_4M 
# ./tests/coccl-tests/build/alltoall_comp_perf -b 8M -e 8G -f 2 -t 8 -g 1 -w 20 -n 100 -c 0

# echo '------------------------------------------------------alltoall comp overlap perf------------------------------------------------------'
# # nsys profile --force-overwrite true -o alltoall_comp_overlap_8G 
# ./tests/coccl-tests/build/alltoall_comp_overlap_perf -b 8G -e 8G -f 2 -t 8 -g 1 -w 20 -n 100 -c 0 

# echo '------------------------------------------------------alltoall comp overlap new perf------------------------------------------------------'
# # nsys profile --force-overwrite true -o alltoall_comp_overlap_new_8M 
# ./tests/coccl-tests/build/alltoall_comp_overlap_new_perf -b 8M -e 8G -f 2 -t 8 -g 1 -w 20 -n 100 -c 0 -G 1

# echo '------------------------------------------------------alltoall comp overlap newnew perf------------------------------------------------------'
# # nsys profile --force-overwrite true -o alltoall_comp_overlap_overlap_1G 
# ./tests/coccl-tests/build/alltoall_comp_overlap_overlap_perf -b 8M -e 4G -f 2 -t 8 -g 1 -w 20 -n 100 -c 0 -G 1

# echo '------------------------------------------------------tahquant allreduce perf------------------------------------------------------'
# ./tests/coccl-tests/build/reduce_scatter_comp_oneshot_perf -b 4M -e 4M -f 2 -t 8 -g 1 -w 0 -n 1 -c 1
# echo '------------------------------------------------------perf------------------------------------------------------'
# ./tests/coccl-tests/build/all_reduce_perf -b 512K -e 4G -f 2 -t 8 -g 1 -z 0 -p 1
# echo '------------------------------------------------------perf------------------------------------------------------'
# ./tests/coccl-tests/build/alltoall_perf -b 512K -e 4G -f 2 -t 8 -g 1
# nsys profile -o alltoall_comp_test ./tests/coccl-tests/build/alltoall_comp_perf -b 32M -e 32M -f 2 -t 8 -g 1 -w 5 -n 10
# echo '------------------------------------------------------comp_perf------------------------------------------------------'
# ./tests/coccl-tests/build/all_gather_comp_perf -b 32K -e 1G -f 2 -t 8 -g 1 -w 20 -n 100
