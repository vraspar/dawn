SKIP: Wave ops not support before SM6.0

RWByteAddressBuffer prevent_dce : register(u0);

vector<float16_t, 4> subgroupMin_cd3b9d() {
  vector<float16_t, 4> arg_0 = (float16_t(1.0h)).xxxx;
  vector<float16_t, 4> res = WaveActiveMin(arg_0);
  return res;
}

[numthreads(1, 1, 1)]
void compute_main() {
  prevent_dce.Store<vector<float16_t, 4> >(0u, subgroupMin_cd3b9d());
  return;
}
