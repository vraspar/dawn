SKIP: FAILED

#version 310 es
precision highp float;
precision highp int;


struct VertexOutput {
  vec4 pos;
  uint prevent_dce;
};

uint prevent_dce;
uint countOneBits_ae44f9() {
  uint arg_0 = 1u;
  uint res = bitCount(arg_0);
  return res;
}
void main() {
  prevent_dce = countOneBits_ae44f9();
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  prevent_dce = countOneBits_ae44f9();
}
VertexOutput main() {
  VertexOutput tint_symbol = VertexOutput(vec4(0.0f), 0u);
  tint_symbol.pos = vec4(0.0f);
  tint_symbol.prevent_dce = countOneBits_ae44f9();
  return tint_symbol;
}
error: Error parsing GLSL shader:
ERROR: 0:14: '=' :  cannot convert from ' global lowp int' to ' temp highp uint'
ERROR: 0:14: '' : compilation terminated 
ERROR: 2 compilation errors.  No code generated.



#version 310 es
precision highp float;
precision highp int;


struct VertexOutput {
  vec4 pos;
  uint prevent_dce;
};

uint prevent_dce;
uint countOneBits_ae44f9() {
  uint arg_0 = 1u;
  uint res = bitCount(arg_0);
  return res;
}
void main() {
  prevent_dce = countOneBits_ae44f9();
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  prevent_dce = countOneBits_ae44f9();
}
VertexOutput main() {
  VertexOutput tint_symbol = VertexOutput(vec4(0.0f), 0u);
  tint_symbol.pos = vec4(0.0f);
  tint_symbol.prevent_dce = countOneBits_ae44f9();
  return tint_symbol;
}
error: Error parsing GLSL shader:
ERROR: 0:14: '=' :  cannot convert from ' global lowp int' to ' temp highp uint'
ERROR: 0:14: '' : compilation terminated 
ERROR: 2 compilation errors.  No code generated.



#version 310 es
precision highp float;
precision highp int;


struct VertexOutput {
  vec4 pos;
  uint prevent_dce;
};

uint prevent_dce;
uint countOneBits_ae44f9() {
  uint arg_0 = 1u;
  uint res = bitCount(arg_0);
  return res;
}
void main() {
  prevent_dce = countOneBits_ae44f9();
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  prevent_dce = countOneBits_ae44f9();
}
VertexOutput main() {
  VertexOutput tint_symbol = VertexOutput(vec4(0.0f), 0u);
  tint_symbol.pos = vec4(0.0f);
  tint_symbol.prevent_dce = countOneBits_ae44f9();
  return tint_symbol;
}
error: Error parsing GLSL shader:
ERROR: 0:14: '=' :  cannot convert from ' global lowp int' to ' temp highp uint'
ERROR: 0:14: '' : compilation terminated 
ERROR: 2 compilation errors.  No code generated.




tint executable returned error: exit status 1
