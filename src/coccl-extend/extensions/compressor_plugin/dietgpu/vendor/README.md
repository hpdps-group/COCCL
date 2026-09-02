# Vendored sources

- `dietgpu/`: `facebookresearch/dietgpu` at commit
  `a4d70a14066d2c3e5fe1849b3723e4cd423eee7e` (MIT). The ANS codec,
  floating-point codec, and their required utilities are included.
- `glog/`: `google/glog` at commit
  `6434410145ee40e7ffe32f97ab54d24d25f0459d` (BSD 3-Clause).

COCCL adds uniform stride entry points and masked stride decoders so Raw and
encoded frames can share one batch, fixes the batched prefix-sum scratch-size
result to report bytes for every batch member, and makes externally supplied
scratch stacks reject overflow instead of calling `cudaMalloc`. The ANS and
floating-point encoded formats are unchanged.
