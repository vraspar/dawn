SKIP: FAILED


RWByteAddressBuffer prevent_dce : register(u0);
float3 subgroupBroadcast_912ff5() {
  float3 arg_0 = (1.0f).xxx;
  float3 res = WaveReadLaneAt(arg_0, 1u);
  return res;
}

[numthreads(1, 1, 1)]
void compute_main() {
  prevent_dce.Store3(0u, asuint(subgroupBroadcast_912ff5()));
}

FXC validation failure:
c:\src\dawn\Shader@0x000001D37D637DD0(5,16-40): error X3004: undeclared identifier 'WaveReadLaneAt'

