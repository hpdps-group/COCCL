```shell
cd coccl/examples/benchmarks_scripts

# for single node
bash communication_benchmarks_single_node.sh <path to cuda> <path to mpi> <path to coccl> <ngpus>
# for multi nodes
bash communication_benchmarks_multi_nodes.sh <path to cuda> <path to mpi> <path to coccl> <gpus_per_node> <nnodes> <path to hostfile>
```

For multi-node runs, `<path to hostfile>` is passed to `mpirun` as `--hostfile`.
