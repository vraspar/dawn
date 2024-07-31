struct VertexOutput {
  float4 pos;
  uint prevent_dce;
};

struct vertex_main_outputs {
  nointerpolation uint VertexOutput_prevent_dce : TEXCOORD0;
  float4 VertexOutput_pos : SV_Position;
};


RWByteAddressBuffer prevent_dce : register(u0);
uint firstLeadingBit_f0779d() {
  uint arg_0 = 1u;
  uint v = arg_0;
  uint v_1 = ((((v & 4294901760u) == 0u)) ? (0u) : (16u));
  uint v_2 = (((((v >> v_1) & 65280u) == 0u)) ? (0u) : (8u));
  uint v_3 = ((((((v >> v_1) >> v_2) & 240u) == 0u)) ? (0u) : (4u));
  uint v_4 = (((((((v >> v_1) >> v_2) >> v_3) & 12u) == 0u)) ? (0u) : (2u));
  uint res = (((((((v >> v_1) >> v_2) >> v_3) >> v_4) == 0u)) ? (4294967295u) : ((v_1 | (v_2 | (v_3 | (v_4 | ((((((((v >> v_1) >> v_2) >> v_3) >> v_4) & 2u) == 0u)) ? (0u) : (1u))))))));
  return res;
}

void fragment_main() {
  prevent_dce.Store(0u, firstLeadingBit_f0779d());
}

[numthreads(1, 1, 1)]
void compute_main() {
  prevent_dce.Store(0u, firstLeadingBit_f0779d());
}

VertexOutput vertex_main_inner() {
  VertexOutput tint_symbol = (VertexOutput)0;
  tint_symbol.pos = (0.0f).xxxx;
  tint_symbol.prevent_dce = firstLeadingBit_f0779d();
  VertexOutput v_5 = tint_symbol;
  return v_5;
}

vertex_main_outputs vertex_main() {
  VertexOutput v_6 = vertex_main_inner();
  VertexOutput v_7 = v_6;
  VertexOutput v_8 = v_6;
  vertex_main_outputs v_9 = {v_8.prevent_dce, v_7.pos};
  return v_9;
}

