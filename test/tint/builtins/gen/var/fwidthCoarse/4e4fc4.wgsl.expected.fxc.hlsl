float4 fwidthCoarse_4e4fc4() {
  float4 arg_0 = (1.0f).xxxx;
  float4 res = fwidth(arg_0);
  return res;
}

RWByteAddressBuffer prevent_dce : register(u0);

void fragment_main() {
  prevent_dce.Store4(0u, asuint(fwidthCoarse_4e4fc4()));
  return;
}
