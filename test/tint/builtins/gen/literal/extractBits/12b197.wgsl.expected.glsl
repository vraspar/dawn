#version 310 es
precision highp float;
precision highp int;

uvec3 extractBits_12b197() {
  uvec3 res = uvec3(0u);
  return res;
}

layout(binding = 0, std430) buffer prevent_dce_block_ssbo {
  uvec3 inner;
  uint pad;
} prevent_dce;

struct VertexOutput {
  vec4 pos;
  uvec3 prevent_dce;
};

void fragment_main() {
  prevent_dce.inner = extractBits_12b197();
}

void main() {
  fragment_main();
  return;
}
#version 310 es

uvec3 extractBits_12b197() {
  uvec3 res = uvec3(0u);
  return res;
}

layout(binding = 0, std430) buffer prevent_dce_block_ssbo {
  uvec3 inner;
  uint pad;
} prevent_dce;

struct VertexOutput {
  vec4 pos;
  uvec3 prevent_dce;
};

void compute_main() {
  prevent_dce.inner = extractBits_12b197();
}

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  compute_main();
  return;
}
#version 310 es

layout(location = 0) flat out uvec3 prevent_dce_1;
uvec3 extractBits_12b197() {
  uvec3 res = uvec3(0u);
  return res;
}

struct VertexOutput {
  vec4 pos;
  uvec3 prevent_dce;
};

VertexOutput vertex_main() {
  VertexOutput tint_symbol = VertexOutput(vec4(0.0f, 0.0f, 0.0f, 0.0f), uvec3(0u, 0u, 0u));
  tint_symbol.pos = vec4(0.0f);
  tint_symbol.prevent_dce = extractBits_12b197();
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
