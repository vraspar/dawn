float dpdxCoarse_029152() {
  float arg_0 = 1.0f;
  float res = ddx_coarse(arg_0);
  return res;
}

RWByteAddressBuffer prevent_dce : register(u0);

void fragment_main() {
  prevent_dce.Store(0u, asuint(dpdxCoarse_029152()));
  return;
}
