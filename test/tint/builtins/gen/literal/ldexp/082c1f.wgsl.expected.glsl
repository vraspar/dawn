#version 310 es
#extension GL_AMD_gpu_shader_half_float : require
precision highp float;
precision highp int;

float16_t ldexp_082c1f() {
  float16_t res = 2.0hf;
  return res;
}

layout(binding = 0, std430) buffer prevent_dce_block_ssbo {
  float16_t inner;
} prevent_dce;

struct VertexOutput {
  vec4 pos;
  float16_t prevent_dce;
};

void fragment_main() {
  prevent_dce.inner = ldexp_082c1f();
}

void main() {
  fragment_main();
  return;
}
#version 310 es
#extension GL_AMD_gpu_shader_half_float : require

float16_t ldexp_082c1f() {
  float16_t res = 2.0hf;
  return res;
}

layout(binding = 0, std430) buffer prevent_dce_block_ssbo {
  float16_t inner;
} prevent_dce;

struct VertexOutput {
  vec4 pos;
  float16_t prevent_dce;
};

void compute_main() {
  prevent_dce.inner = ldexp_082c1f();
}

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  compute_main();
  return;
}
#version 310 es
#extension GL_AMD_gpu_shader_half_float : require

layout(location = 0) flat out float16_t prevent_dce_1;
float16_t ldexp_082c1f() {
  float16_t res = 2.0hf;
  return res;
}

struct VertexOutput {
  vec4 pos;
  float16_t prevent_dce;
};

VertexOutput vertex_main() {
  VertexOutput tint_symbol = VertexOutput(vec4(0.0f, 0.0f, 0.0f, 0.0f), 0.0hf);
  tint_symbol.pos = vec4(0.0f);
  tint_symbol.prevent_dce = ldexp_082c1f();
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
