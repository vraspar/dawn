float4 dpdyFine_d0a648() {
  float4 res = ddy_fine((1.0f).xxxx);
  return res;
}

RWByteAddressBuffer prevent_dce : register(u0);

void fragment_main() {
  prevent_dce.Store4(0u, asuint(dpdyFine_d0a648()));
  return;
}
