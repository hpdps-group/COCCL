```shell
git clone https://github.com/hpdps-group/coccl.git
chmod 777 -R coccl
cd coccl
bash build.sh <path to cuda> \
<path to mpi> \
<path to coccl> \
"-gencode=arch=compute_90,code=sm_90"
# use the corresponding NVCC_GENCODE for your hardware
```
