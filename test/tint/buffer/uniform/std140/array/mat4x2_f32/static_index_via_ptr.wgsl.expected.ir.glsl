#version 310 es


struct mat4x2_f32_std140 {
  vec2 col0;
  vec2 col1;
  vec2 col2;
  vec2 col3;
};

layout(binding = 0, std140)
uniform tint_symbol_1_std140_1_ubo {
  mat4x2_f32_std140 tint_symbol[4];
} v;
layout(binding = 1, std430)
buffer tint_symbol_3_1_ssbo {
  float tint_symbol_2;
} v_1;
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  mat4x2 v_2 = mat4x2(v.tint_symbol[2].col0, v.tint_symbol[2].col1, v.tint_symbol[2].col2, v.tint_symbol[2].col3);
  mat4x2_f32_std140 v_3[4] = v.tint_symbol;
  mat4x2 v_4[4] = mat4x2[4](mat4x2(vec2(0.0f), vec2(0.0f), vec2(0.0f), vec2(0.0f)), mat4x2(vec2(0.0f), vec2(0.0f), vec2(0.0f), vec2(0.0f)), mat4x2(vec2(0.0f), vec2(0.0f), vec2(0.0f), vec2(0.0f)), mat4x2(vec2(0.0f), vec2(0.0f), vec2(0.0f), vec2(0.0f)));
  {
    uint v_5 = 0u;
    v_5 = 0u;
    while(true) {
      uint v_6 = v_5;
      if ((v_6 >= 4u)) {
        break;
      }
      v_4[v_6] = mat4x2(v_3[v_6].col0, v_3[v_6].col1, v_3[v_6].col2, v_3[v_6].col3);
      {
        v_5 = (v_6 + 1u);
      }
      continue;
    }
  }
  mat4x2 l_a[4] = v_4;
  mat4x2 l_a_i = v_2;
  vec2 l_a_i_i = v_2[1];
  v_1.tint_symbol_2 = (((v_2[1][0u] + l_a[0][0][0u]) + l_a_i[0][0u]) + l_a_i_i[0u]);
}
