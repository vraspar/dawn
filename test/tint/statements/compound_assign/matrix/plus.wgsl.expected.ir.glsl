#version 310 es


struct S {
  mat4 a;
};

layout(binding = 0, std430)
buffer tint_symbol_1_1_ssbo {
  S tint_symbol;
} v_1;
void foo() {
  v_1.tint_symbol.a = (v_1.tint_symbol.a + mat4(vec4(0.0f), vec4(0.0f), vec4(0.0f), vec4(0.0f)));
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
}
