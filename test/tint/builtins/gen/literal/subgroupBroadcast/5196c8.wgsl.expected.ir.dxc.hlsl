
RWByteAddressBuffer prevent_dce : register(u0);
float2 subgroupBroadcast_5196c8() {
  float2 res = WaveReadLaneAt((1.0f).xx, 1u);
  return res;
}

[numthreads(1, 1, 1)]
void compute_main() {
  prevent_dce.Store2(0u, asuint(subgroupBroadcast_5196c8()));
}

