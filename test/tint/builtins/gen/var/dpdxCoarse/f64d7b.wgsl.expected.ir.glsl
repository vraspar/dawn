#version 310 es
precision highp float;
precision highp int;

layout(binding = 0, std430)
buffer tint_symbol_1_1_ssbo {
  vec3 tint_symbol;
} v;
vec3 dpdxCoarse_f64d7b() {
  vec3 arg_0 = vec3(1.0f);
  vec3 res = dFdx(arg_0);
  return res;
}
void main() {
  v.tint_symbol = dpdxCoarse_f64d7b();
}
