# COCCL_PPoPP_AE

## Build

To build the library :

```shell
git clone https://github.com/maomaohpp/COCCL_PPoPP_AE.git
chmod 777 -R COCCL_PPoPP_AE
cd COCCL_PPoPP_AE
bash build.sh /path/to/cuda \
/path/to/mpi \
/path/to/COCCL_PPoPP_AE \
"-gencode=arch=compute_90,code=sm_90"
# use the corresponding NVCC_GENCODE for your hardware
```

## Intrgrated with Framework

Please install the necessary environment dependencies for training frameworks such as [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) or [PyTorch](https://github.com/pytorch/pytorch).

To switch from the NCCL library to the COCCL library, follow the steps below:

1. Confirm whether the NCCL library used by PyTorch is a dynamic library.

   - Confirm the location of the PyTorch library.
     If you know that PyTorch is installed in a specific directory, you can search directly within that directory. For example, after confirming that PyTorch resides in `/usr/local/lib`, running the query command successfully pinpointed the exact path to the `libtorch.so` file, as shown below:

     ```bash
     find /usr/local/lib -name "libtorch*"
     # The example results are as follows:
     /usr/local/lib/python3.10/dist-packages/torch/lib/libtorchcuda.so
     /usr/local/lib/python3.10/dist-packages/torch/lib/libtorch.so
     /usr/local/lib/python3.10/dist-packages/torch/lib/libtorchbindtest.so
     ```

   - Use the `ldd` command to inspect the PyTorch library’s dependency on the NCCL library.

     ```bash
     ldd libtorch.so | grep nccl
     ```

     If the command returns results in the following format, it indicates that PyTorch depends on NCCL as a dynamic library. You may then proceed to configure COCCL according to the subsequent steps.

     ```bash
     libnccl.so.2=>/usr/lib/x86_64-linux-gnu/libnccl.so.2(0x00007feab3b27000)
     ```

     If the command returns no results, this indicates that PyTorch relies on NCCL as a static (non-dynamic) library and therefore cannot be switched to COCCL. To proceed with COCCL configuration, you must use a PyTorch version that depends on the NCCL dynamic library.

2. Before running the training script, please specify the paths to COCCL and the compressor dynamic libraries in COCCL_PPoPP_AE/training_scripts/training_envs.sh.

   ```bash
   #!/bin/bash
   export CUDA_PATH=/path/to/cuda
   export COCCL_PATH=/path/to/COCCL_PPoPP_AE
   export MEGATRON_PATH=/path/to/Megatron
   export DATASET_PATH=/path/to/dataset
   ```

## Training-Mode Compressor Selection

Training mode infers whether each communicator is used for DP, PP, or TP and
then selects a compressor chain for that role:

```bash
export NCCL_ENABLE_COMPRESS=1
export NCCL_COCCL_TRAINING_MODE=1

# Process-wide plugin allow-list.
export NCCL_COMPRESSORS=sdp4bit,tahquant

# Role switches are independent from role classification.
export NCCL_ENABLE_DP_COMPRESS=1
export NCCL_ENABLE_PP_COMPRESS=1
export NCCL_ENABLE_TP_COMPRESS=1

# Comma-separated role/primitive chains. Every name must also appear above.
export NCCL_DP_ALLGATHER_COMPRESSORS=sdp4bit
export NCCL_DP_REDUCESCATTER_COMPRESSORS=sdp4bit
export NCCL_DP_ALLREDUCE_COMPRESSORS=sdp4bit
export NCCL_DP_ALLGATHER_COMPRESS_ENABLE_THRESHOLD=8388608
export NCCL_DP_REDUCESCATTER_COMPRESS_ENABLE_THRESHOLD=8388608
export NCCL_DP_ALLREDUCE_COMPRESS_ENABLE_THRESHOLD=8388608

export NCCL_TP_ALLGATHER_COMPRESSORS=sdp4bit
export NCCL_TP_REDUCESCATTER_COMPRESSORS=sdp4bit
export NCCL_TP_ALLREDUCE_COMPRESSORS=sdp4bit
export NCCL_TP_ALLGATHER_COMPRESS_ENABLE_THRESHOLD=8388608
export NCCL_TP_REDUCESCATTER_COMPRESS_ENABLE_THRESHOLD=8388608
export NCCL_TP_ALLREDUCE_COMPRESS_ENABLE_THRESHOLD=8388608

export NCCL_PP_SENDRECV_FWD_COMPRESSORS=tahquant
export NCCL_PP_SENDRECV_BWD_COMPRESSORS=tahquant
export NCCL_PP_SENDRECV_COMPRESS_ENABLE_THRESHOLD=8388608
```

In training mode, all primitive-level `NCCL_ENABLE_<OP>_COMPRESS` and
`NCCL_<OP>_COMPRESSORS` settings are ignored. COCCL still classifies every
observed communicator even when its role switch is disabled. Before
classification commits, for an `Unknown` role, or when the corresponding role
switch is disabled, communication stays on native NCCL. A role switch does not
enable all primitives by itself: an enabled role only compresses operations
whose `NCCL_<ROLE>_<OP>_COMPRESSORS` list is non-empty. Missing role/operation
configuration stays on native NCCL and does not fall back to the global chain.
The retired `NCCL_DP_COMPRESSORS`, `NCCL_PP_COMPRESSORS`, and
`NCCL_TP_COMPRESSORS` variables are ignored. Operation config files such as
`sdp4bit_RS.config` remain active for explicitly configured role/op chains.

Compression thresholds are uncompressed logical bytes and default to 8 MiB
(`8388608` bytes).
Normal mode uses `NCCL_<OP>_COMPRESS_ENABLE_THRESHOLD`, where `<OP>` is
`ALLTOALL`, `ALLREDUCE`, `ALLGATHER`, `REDUCESCATTER`, or `SENDRECV`. Training
mode uses the role/operation variables shown above. Internal `*_Inter` stages
reuse their parent collective threshold, and forward/backward Send/Recv share
one SendRecv threshold. The former global `NCCL_COMPRESS_ENABLE_THRESHOLD` is
ignored.
