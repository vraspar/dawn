SKIP: Wave ops not support before SM6.0

RWByteAddressBuffer prevent_dce : register(u0);

float subgroupMax_1a1a5f() {
  float res = WaveActiveMax(1.0f);
  return res;
}

[numthreads(1, 1, 1)]
void compute_main() {
  prevent_dce.Store(0u, asuint(subgroupMax_1a1a5f()));
  return;
}
