RWByteAddressBuffer prevent_dce : register(u0);

int subgroupBroadcast_1d79c7() {
  int res = WaveReadLaneAt(1, 1u);
  return res;
}

void fragment_main() {
  prevent_dce.Store(0u, asuint(subgroupBroadcast_1d79c7()));
  return;
}

[numthreads(1, 1, 1)]
void compute_main() {
  prevent_dce.Store(0u, asuint(subgroupBroadcast_1d79c7()));
  return;
}
