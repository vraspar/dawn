#version 310 es

layout(binding = 0, std430)
buffer tint_symbol_3_1_ssbo {
  uint tint_symbol_2;
} v_1;
uint foo() {
  return uint[4](0u, 1u, 2u, 4u)[v_1.tint_symbol_2];
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  uint v = uint[4](0u, 1u, 2u, 4u)[v_1.tint_symbol_2];
  v_1.tint_symbol_2 = (v + foo());
}
