SKIP: FAILED

#version 310 es

struct buf0 {
  float two;
};

struct main_out {
  vec4 x_GLF_color_1;
};
precision highp float;
precision highp int;


uniform buf0 x_7;
vec4 x_GLF_color = vec4(0.0f);
float func_vf2_(inout vec2 v) {
  float x_42 = x_7.two;
  v[0u] = x_42;
  float x_45 = v.y;
  if ((x_45 < 1.0f)) {
    return 1.0f;
  }
  return 5.0f;
}
void main_1() {
  float f = 0.0f;
  vec2 param = vec2(0.0f);
  param = vec2(1.0f);
  float x_34 = func_vf2_(param);
  f = x_34;
  float x_35 = f;
  if ((x_35 == 5.0f)) {
    x_GLF_color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
  } else {
    x_GLF_color = vec4(0.0f);
  }
}
main_out main() {
  main_1();
  return main_out(x_GLF_color);
}
error: Error parsing GLSL shader:
ERROR: 0:4: 'float' : type requires declaration of default precision qualifier 
ERROR: 0:4: '' : compilation terminated 
ERROR: 2 compilation errors.  No code generated.




tint executable returned error: exit status 1
