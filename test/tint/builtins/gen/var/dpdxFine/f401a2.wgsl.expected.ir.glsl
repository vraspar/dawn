SKIP: FAILED

#version 310 es
precision highp float;
precision highp int;


float prevent_dce;
float dpdxFine_f401a2() {
  float arg_0 = 1.0f;
  float res = dFdxdFdxFine(arg_0);
  return res;
}
void main() {
  prevent_dce = dpdxFine_f401a2();
}
error: Error parsing GLSL shader:
ERROR: 0:9: 'dFdxdFdxFine' : no matching overloaded function found 
ERROR: 0:9: '' : compilation terminated 
ERROR: 2 compilation errors.  No code generated.




tint executable returned error: exit status 1
