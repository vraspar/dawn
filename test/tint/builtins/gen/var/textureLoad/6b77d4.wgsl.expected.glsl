#version 310 es
precision highp float;
precision highp int;

uniform highp usampler2D arg_0_1;
uvec4 textureLoad_6b77d4() {
  int arg_1 = 1;
  uint arg_2 = 1u;
  uvec4 res = texelFetch(arg_0_1, ivec2(arg_1, 0), int(arg_2));
  return res;
}

layout(binding = 0, std430) buffer prevent_dce_block_ssbo {
  uvec4 inner;
} prevent_dce;

struct VertexOutput {
  vec4 pos;
  uvec4 prevent_dce;
};

void fragment_main() {
  prevent_dce.inner = textureLoad_6b77d4();
}

void main() {
  fragment_main();
  return;
}
#version 310 es

uniform highp usampler2D arg_0_1;
uvec4 textureLoad_6b77d4() {
  int arg_1 = 1;
  uint arg_2 = 1u;
  uvec4 res = texelFetch(arg_0_1, ivec2(arg_1, 0), int(arg_2));
  return res;
}

layout(binding = 0, std430) buffer prevent_dce_block_ssbo {
  uvec4 inner;
} prevent_dce;

struct VertexOutput {
  vec4 pos;
  uvec4 prevent_dce;
};

void compute_main() {
  prevent_dce.inner = textureLoad_6b77d4();
}

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  compute_main();
  return;
}
#version 310 es

layout(location = 0) flat out uvec4 prevent_dce_1;
uniform highp usampler2D arg_0_1;
uvec4 textureLoad_6b77d4() {
  int arg_1 = 1;
  uint arg_2 = 1u;
  uvec4 res = texelFetch(arg_0_1, ivec2(arg_1, 0), int(arg_2));
  return res;
}

struct VertexOutput {
  vec4 pos;
  uvec4 prevent_dce;
};

VertexOutput vertex_main() {
  VertexOutput tint_symbol = VertexOutput(vec4(0.0f, 0.0f, 0.0f, 0.0f), uvec4(0u, 0u, 0u, 0u));
  tint_symbol.pos = vec4(0.0f);
  tint_symbol.prevent_dce = textureLoad_6b77d4();
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
