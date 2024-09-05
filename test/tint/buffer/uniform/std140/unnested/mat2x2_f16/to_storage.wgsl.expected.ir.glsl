#version 310 es
#extension GL_AMD_gpu_shader_half_float: require

layout(binding = 0, std140)
uniform tint_symbol_1_std140_1_ubo {
  f16vec2 tint_symbol_col0;
  f16vec2 tint_symbol_col1;
} v;
layout(binding = 1, std430)
buffer tint_symbol_3_1_ssbo {
  f16mat2 tint_symbol_2;
} v_1;
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  v_1.tint_symbol_2 = f16mat2(v.tint_symbol_col0, v.tint_symbol_col1);
  v_1.tint_symbol_2[1] = f16mat2(v.tint_symbol_col0, v.tint_symbol_col1)[0];
  v_1.tint_symbol_2[1] = f16mat2(v.tint_symbol_col0, v.tint_symbol_col1)[0].yx;
  v_1.tint_symbol_2[0][1] = f16mat2(v.tint_symbol_col0, v.tint_symbol_col1)[1][0];
}
