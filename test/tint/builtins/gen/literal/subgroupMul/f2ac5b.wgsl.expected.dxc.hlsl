RWByteAddressBuffer prevent_dce : register(u0);

vector<float16_t, 4> subgroupMul_f2ac5b() {
  vector<float16_t, 4> res = WaveActiveProduct((float16_t(1.0h)).xxxx);
  return res;
}

void fragment_main() {
  prevent_dce.Store<vector<float16_t, 4> >(0u, subgroupMul_f2ac5b());
  return;
}

[numthreads(1, 1, 1)]
void compute_main() {
  prevent_dce.Store<vector<float16_t, 4> >(0u, subgroupMul_f2ac5b());
  return;
}
