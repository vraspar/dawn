
RWByteAddressBuffer prevent_dce : register(u0);
float subgroupMin_7def0a() {
  float arg_0 = 1.0f;
  float res = WaveActiveMin(arg_0);
  return res;
}

void fragment_main() {
  prevent_dce.Store(0u, asuint(subgroupMin_7def0a()));
}

[numthreads(1, 1, 1)]
void compute_main() {
  prevent_dce.Store(0u, asuint(subgroupMin_7def0a()));
}

