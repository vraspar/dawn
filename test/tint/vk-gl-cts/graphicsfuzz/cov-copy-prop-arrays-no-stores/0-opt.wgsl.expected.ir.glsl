SKIP: FAILED

#version 310 es

struct buf0 {
  int zero;
};

struct Array {
  int values[2];
};

struct main_out {
  vec4 x_GLF_color_1;
};
precision highp float;
precision highp int;


uniform buf0 x_7;
vec4 x_GLF_color = vec4(0.0f);
void main_1() {
  Array a = Array(int[2](0, 0));
  Array b = Array(int[2](0, 0));
  float one = 0.0f;
  int x_10 = x_7.zero;
  a.values[x_10] = 1;
  Array x_35 = a;
  b = x_35;
  one = 0.0f;
  int x_11 = x_7.zero;
  int x_12 = b.values[x_11];
  if ((x_12 == 1)) {
    one = 1.0f;
  }
  float x_41 = one;
  x_GLF_color = vec4(x_41, 0.0f, 0.0f, 1.0f);
}
main_out main() {
  main_1();
  return main_out(x_GLF_color);
}
error: Error parsing GLSL shader:
ERROR: 0:12: 'float' : type requires declaration of default precision qualifier 
ERROR: 0:12: '' : compilation terminated 
ERROR: 2 compilation errors.  No code generated.




tint executable returned error: exit status 1
