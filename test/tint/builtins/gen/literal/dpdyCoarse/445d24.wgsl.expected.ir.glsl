SKIP: FAILED

#version 310 es
precision highp float;
precision highp int;


vec4 prevent_dce;
vec4 dpdyCoarse_445d24() {
  vec4 res = dFdydFdyCoarse(vec4(1.0f));
  return res;
}
void main() {
  prevent_dce = dpdyCoarse_445d24();
}
error: Error parsing GLSL shader:
ERROR: 0:8: 'dFdydFdyCoarse' : no matching overloaded function found 
ERROR: 0:8: '=' :  cannot convert from ' const float' to ' temp highp 4-component vector of float'
ERROR: 0:8: '' : compilation terminated 
ERROR: 3 compilation errors.  No code generated.




tint executable returned error: exit status 1
