struct VertexOutput {
  float4 pos;
  int4 prevent_dce;
};

struct vertex_main_outputs {
  nointerpolation int4 VertexOutput_prevent_dce : TEXCOORD0;
  float4 VertexOutput_pos : SV_Position;
};


RWByteAddressBuffer prevent_dce : register(u0);
Texture3D<int4> arg_0 : register(t0, space1);
int4 textureLoad_4fd803() {
  int3 arg_1 = (1).xxx;
  int arg_2 = 1;
  Texture3D<int4> v = arg_0;
  int v_1 = arg_2;
  int3 v_2 = int3(arg_1);
  int4 res = int4(v.Load(int4(v_2, int(v_1))));
  return res;
}

void fragment_main() {
  prevent_dce.Store4(0u, asuint(textureLoad_4fd803()));
}

[numthreads(1, 1, 1)]
void compute_main() {
  prevent_dce.Store4(0u, asuint(textureLoad_4fd803()));
}

VertexOutput vertex_main_inner() {
  VertexOutput tint_symbol = (VertexOutput)0;
  tint_symbol.pos = (0.0f).xxxx;
  tint_symbol.prevent_dce = textureLoad_4fd803();
  VertexOutput v_3 = tint_symbol;
  return v_3;
}

vertex_main_outputs vertex_main() {
  VertexOutput v_4 = vertex_main_inner();
  VertexOutput v_5 = v_4;
  VertexOutput v_6 = v_4;
  vertex_main_outputs v_7 = {v_6.prevent_dce, v_5.pos};
  return v_7;
}

