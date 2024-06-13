#version 310 es
precision highp float;
precision highp int;

ivec4 tint_insert_bits(ivec4 v, ivec4 n, uint offset, uint count) {
  uint s = min(offset, 32u);
  uint e = min(32u, (s + count));
  return bitfieldInsert(v, n, int(s), int((e - s)));
}

ivec4 insertBits_d86978() {
  ivec4 arg_0 = ivec4(1);
  ivec4 arg_1 = ivec4(1);
  uint arg_2 = 1u;
  uint arg_3 = 1u;
  ivec4 res = tint_insert_bits(arg_0, arg_1, arg_2, arg_3);
  return res;
}

layout(binding = 0, std430) buffer prevent_dce_block_ssbo {
  ivec4 inner;
} prevent_dce;

struct VertexOutput {
  vec4 pos;
  ivec4 prevent_dce;
};

void fragment_main() {
  prevent_dce.inner = insertBits_d86978();
}

void main() {
  fragment_main();
  return;
}
#version 310 es

ivec4 tint_insert_bits(ivec4 v, ivec4 n, uint offset, uint count) {
  uint s = min(offset, 32u);
  uint e = min(32u, (s + count));
  return bitfieldInsert(v, n, int(s), int((e - s)));
}

ivec4 insertBits_d86978() {
  ivec4 arg_0 = ivec4(1);
  ivec4 arg_1 = ivec4(1);
  uint arg_2 = 1u;
  uint arg_3 = 1u;
  ivec4 res = tint_insert_bits(arg_0, arg_1, arg_2, arg_3);
  return res;
}

layout(binding = 0, std430) buffer prevent_dce_block_ssbo {
  ivec4 inner;
} prevent_dce;

struct VertexOutput {
  vec4 pos;
  ivec4 prevent_dce;
};

void compute_main() {
  prevent_dce.inner = insertBits_d86978();
}

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  compute_main();
  return;
}
#version 310 es

ivec4 tint_insert_bits(ivec4 v, ivec4 n, uint offset, uint count) {
  uint s = min(offset, 32u);
  uint e = min(32u, (s + count));
  return bitfieldInsert(v, n, int(s), int((e - s)));
}

layout(location = 0) flat out ivec4 prevent_dce_1;
ivec4 insertBits_d86978() {
  ivec4 arg_0 = ivec4(1);
  ivec4 arg_1 = ivec4(1);
  uint arg_2 = 1u;
  uint arg_3 = 1u;
  ivec4 res = tint_insert_bits(arg_0, arg_1, arg_2, arg_3);
  return res;
}

struct VertexOutput {
  vec4 pos;
  ivec4 prevent_dce;
};

VertexOutput vertex_main() {
  VertexOutput tint_symbol = VertexOutput(vec4(0.0f, 0.0f, 0.0f, 0.0f), ivec4(0, 0, 0, 0));
  tint_symbol.pos = vec4(0.0f);
  tint_symbol.prevent_dce = insertBits_d86978();
  return tint_symbol;
}

void main() {
  gl_PointSize = 1.0;
  VertexOutput inner_result = vertex_main();
  gl_Position = inner_result.pos;
  prevent_dce_1 = inner_result.prevent_dce;
  gl_Position.y = -(gl_Position.y);
  gl_Position.z = ((2.0f * gl_Position.z) - gl_Position.w);
  return;
}
