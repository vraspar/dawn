#version 310 es

layout(binding = 0, std430)
buffer tint_symbol_2_1_ssbo {
  vec4 tint_symbol_1;
} v;
void func() {
  v.tint_symbol_1 = vec4(0.0f);
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  func();
}
