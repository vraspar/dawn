SKIP: FAILED

#version 310 es

struct PixelLocal {
  uint a;
  int b;
  float c;
};
precision highp float;
precision highp int;


PixelLocal P;
vec4 main() {
  P.a = (P.a + 42u);
  return vec4(2.0f);
}
error: Error parsing GLSL shader:
ERROR: 0:6: 'float' : type requires declaration of default precision qualifier 
ERROR: 0:6: '' : compilation terminated 
ERROR: 2 compilation errors.  No code generated.




tint executable returned error: exit status 1
