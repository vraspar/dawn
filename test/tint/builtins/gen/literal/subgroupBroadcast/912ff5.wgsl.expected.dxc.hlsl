float3 subgroupBroadcast_912ff5() {
  float3 res = WaveReadLaneAt((1.0f).xxx, 1u);
  return res;
}

RWByteAddressBuffer prevent_dce : register(u0);

[numthreads(1, 1, 1)]
void compute_main() {
  prevent_dce.Store3(0u, asuint(subgroupBroadcast_912ff5()));
  return;
}
