SKIP: FAILED

#version 310 es

struct strided_arr {
  float el;
};

struct buf0 {
  strided_arr x_GLF_uniform_float_values[2];
};

struct main_out {
  vec4 x_GLF_color_1;
};
precision highp float;
precision highp int;


vec4 x_GLF_color = vec4(0.0f);
uniform buf0 x_8;
void main_1() {
  float a = 0.0f;
  float b = 0.0f;
  float c = 0.0f;
  a = -1.0f;
  b = 1.70000004768371582031f;
  float x_27 = a;
  float x_28 = b;
  c = pow(x_27, x_28);
  float x_30 = c;
  x_GLF_color = vec4(x_30, x_30, x_30, x_30);
  float x_32 = a;
  float x_34 = b;
  if (((x_32 == -1.0f) & (x_34 == 1.70000004768371582031f))) {
    float x_41 = x_8.x_GLF_uniform_float_values[0].el;
    float x_43 = x_8.x_GLF_uniform_float_values[1].el;
    float x_45 = x_8.x_GLF_uniform_float_values[1].el;
    float x_47 = x_8.x_GLF_uniform_float_values[0].el;
    x_GLF_color = vec4(x_41, x_43, x_45, x_47);
  } else {
    float x_50 = x_8.x_GLF_uniform_float_values[0].el;
    x_GLF_color = vec4(x_50, x_50, x_50, x_50);
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
