SKIP: Wave ops not support before SM6.0

RWByteAddressBuffer prevent_dce : register(u0);

uint4 subgroupMax_15ccbf() {
  uint4 arg_0 = (1u).xxxx;
  uint4 res = WaveActiveMax(arg_0);
  return res;
}

[numthreads(1, 1, 1)]
void compute_main() {
  prevent_dce.Store4(0u, asuint(subgroupMax_15ccbf()));
  return;
}
