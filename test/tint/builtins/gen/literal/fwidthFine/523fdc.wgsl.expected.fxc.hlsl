float3 tint_fwidth_fine(float3 v) {
  return (abs(ddx_fine(v)) + abs(ddy_fine(v)));
}

float3 fwidthFine_523fdc() {
  float3 res = tint_fwidth_fine((1.0f).xxx);
  return res;
}

RWByteAddressBuffer prevent_dce : register(u0);

void fragment_main() {
  prevent_dce.Store3(0u, asuint(fwidthFine_523fdc()));
  return;
}
