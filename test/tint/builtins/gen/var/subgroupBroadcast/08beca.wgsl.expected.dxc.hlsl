float subgroupBroadcast_08beca() {
  float arg_0 = 1.0f;
  float res = WaveReadLaneAt(arg_0, 1u);
  return res;
}

RWByteAddressBuffer prevent_dce : register(u0);

[numthreads(1, 1, 1)]
void compute_main() {
  prevent_dce.Store(0u, asuint(subgroupBroadcast_08beca()));
  return;
}
