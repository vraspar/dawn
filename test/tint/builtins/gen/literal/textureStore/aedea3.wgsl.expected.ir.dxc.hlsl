
RWTexture2DArray<uint4> arg_0 : register(u0, space1);
void textureStore_aedea3() {
  RWTexture2DArray<uint4> v = arg_0;
  v[uint3((1u).xx, uint(1))] = (1u).xxxx;
}

void fragment_main() {
  textureStore_aedea3();
}

[numthreads(1, 1, 1)]
void compute_main() {
  textureStore_aedea3();
}

