SKIP: FAILED

struct frexp_result_vec2_f32 {
  float2 fract;
  int2 exp;
};

struct VertexOutput {
  float4 pos;
};

struct vertex_main_outputs {
  float4 VertexOutput_pos : SV_Position;
};


void frexp_eb2421() {
  float2 arg_0 = (1.0f).xx;
  frexp_result_vec2_f32 res = frexp(arg_0);
}

void fragment_main() {
  frexp_eb2421();
}

[numthreads(1, 1, 1)]
void compute_main() {
  frexp_eb2421();
}

VertexOutput vertex_main_inner() {
  VertexOutput tint_symbol = (VertexOutput)0;
  tint_symbol.pos = (0.0f).xxxx;
  frexp_eb2421();
  VertexOutput v = tint_symbol;
  return v;
}

vertex_main_outputs vertex_main() {
  VertexOutput v_1 = vertex_main_inner();
  vertex_main_outputs v_2 = {v_1.pos};
  return v_2;
}

DXC validation failure:
hlsl.hlsl:17:31: error: use of undeclared identifier 'frexp'
  frexp_result_vec2_f32 res = frexp(arg_0);
                              ^

