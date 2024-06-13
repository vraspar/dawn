#version 310 es
precision highp float;
precision highp int;

vec3 smoothstep_aad1db() {
  vec3 arg_0 = vec3(2.0f);
  vec3 arg_1 = vec3(4.0f);
  vec3 arg_2 = vec3(3.0f);
  vec3 res = smoothstep(arg_0, arg_1, arg_2);
  return res;
}

layout(binding = 0, std430) buffer prevent_dce_block_ssbo {
  vec3 inner;
  uint pad;
} prevent_dce;

struct VertexOutput {
  vec4 pos;
  vec3 prevent_dce;
};

void fragment_main() {
  prevent_dce.inner = smoothstep_aad1db();
}

void main() {
  fragment_main();
  return;
}
#version 310 es

vec3 smoothstep_aad1db() {
  vec3 arg_0 = vec3(2.0f);
  vec3 arg_1 = vec3(4.0f);
  vec3 arg_2 = vec3(3.0f);
  vec3 res = smoothstep(arg_0, arg_1, arg_2);
  return res;
}

layout(binding = 0, std430) buffer prevent_dce_block_ssbo {
  vec3 inner;
  uint pad;
} prevent_dce;

struct VertexOutput {
  vec4 pos;
  vec3 prevent_dce;
};

void compute_main() {
  prevent_dce.inner = smoothstep_aad1db();
}

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  compute_main();
  return;
}
#version 310 es

layout(location = 0) flat out vec3 prevent_dce_1;
vec3 smoothstep_aad1db() {
  vec3 arg_0 = vec3(2.0f);
  vec3 arg_1 = vec3(4.0f);
  vec3 arg_2 = vec3(3.0f);
  vec3 res = smoothstep(arg_0, arg_1, arg_2);
  return res;
}

struct VertexOutput {
  vec4 pos;
  vec3 prevent_dce;
};

VertexOutput vertex_main() {
  VertexOutput tint_symbol = VertexOutput(vec4(0.0f, 0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.0f));
  tint_symbol.pos = vec4(0.0f);
  tint_symbol.prevent_dce = smoothstep_aad1db();
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
