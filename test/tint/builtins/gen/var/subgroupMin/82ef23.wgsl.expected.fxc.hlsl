SKIP: Wave ops not support before SM6.0

RWByteAddressBuffer prevent_dce : register(u0);

uint4 subgroupMin_82ef23() {
  uint4 arg_0 = (1u).xxxx;
  uint4 res = WaveActiveMin(arg_0);
  return res;
}

[numthreads(1, 1, 1)]
void compute_main() {
  prevent_dce.Store4(0u, asuint(subgroupMin_82ef23()));
  return;
}
