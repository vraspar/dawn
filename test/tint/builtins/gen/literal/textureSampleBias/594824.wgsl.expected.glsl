#version 310 es
precision highp float;
precision highp int;

uniform highp sampler3D arg_0_arg_1;

vec4 textureSampleBias_594824() {
  vec4 res = textureOffset(arg_0_arg_1, vec3(1.0f), ivec3(1), 1.0f);
  return res;
}

layout(binding = 0, std430) buffer prevent_dce_block_ssbo {
  vec4 inner;
} prevent_dce;

void fragment_main() {
  prevent_dce.inner = textureSampleBias_594824();
}

void main() {
  fragment_main();
  return;
}
