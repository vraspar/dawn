#version 310 es
precision highp float;
precision highp int;

layout(binding = 0, std430)
buffer tint_symbol_1_1_ssbo {
  vec4 tint_symbol;
} v;
vec4 dpdy_699a05() {
  vec4 res = dFdy(vec4(1.0f));
  return res;
}
void main() {
  v.tint_symbol = dpdy_699a05();
}
