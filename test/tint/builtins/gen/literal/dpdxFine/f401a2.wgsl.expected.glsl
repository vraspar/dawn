#version 310 es
precision highp float;
precision highp int;

float dpdxFine_f401a2() {
  float res = dFdx(1.0f);
  return res;
}

layout(binding = 0, std430) buffer prevent_dce_block_ssbo {
  float inner;
} prevent_dce;

void fragment_main() {
  prevent_dce.inner = dpdxFine_f401a2();
}

void main() {
  fragment_main();
  return;
}
