
RWTexture2D<int4> arg_0 : register(u0, space1);
void textureStore_752da6() {
  int2 arg_1 = (1).xx;
  int4 arg_2 = (1).xxxx;
  arg_0[arg_1] = arg_2;
}

void fragment_main() {
  textureStore_752da6();
}

[numthreads(1, 1, 1)]
void compute_main() {
  textureStore_752da6();
}

