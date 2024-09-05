#version 310 es
#extension GL_AMD_gpu_shader_half_float: require

layout(binding = 0, std140)
uniform tint_symbol_2_std140_1_ubo {
  f16vec3 tint_symbol_1_col0;
  f16vec3 tint_symbol_1_col1;
  f16vec3 tint_symbol_1_col2;
  f16vec3 tint_symbol_1_col3;
} v;
layout(binding = 1, std430)
buffer tint_symbol_4_1_ssbo {
  f16mat4x3 tint_symbol_3;
} v_1;
void tint_store_and_preserve_padding(inout f16mat4x3 target, f16mat4x3 value_param) {
  target[0u] = value_param[0u];
  target[1u] = value_param[1u];
  target[2u] = value_param[2u];
  target[3u] = value_param[3u];
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  f16mat4x3 x = f16mat4x3(v.tint_symbol_1_col0, v.tint_symbol_1_col1, v.tint_symbol_1_col2, v.tint_symbol_1_col3);
  tint_store_and_preserve_padding(v_1.tint_symbol_3, x);
}
