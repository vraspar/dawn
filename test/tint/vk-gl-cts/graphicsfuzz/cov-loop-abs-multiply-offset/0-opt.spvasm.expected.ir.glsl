SKIP: FAILED

#version 310 es

struct strided_arr {
  float el;
};

struct buf0 {
  strided_arr x_GLF_uniform_float_values[4];
};

struct strided_arr_1 {
  int el;
};

struct buf1 {
  strided_arr_1 x_GLF_uniform_int_values[3];
};

struct main_out {
  vec4 x_GLF_color_1;
};
precision highp float;
precision highp int;


uniform buf0 x_6;
uniform buf1 x_9;
vec4 x_GLF_color = vec4(0.0f);
void main_1() {
  float f = 0.0f;
  int i = 0;
  bool x_66 = false;
  bool x_67 = false;
  f = x_6.x_GLF_uniform_float_values[0].el;
  i = x_9.x_GLF_uniform_int_values[1].el;
  {
    while(true) {
      if ((i < x_9.x_GLF_uniform_int_values[0].el)) {
      } else {
        break;
      }
      float v = abs((-(x_6.x_GLF_uniform_float_values[3].el) * f));
      f = (v + x_6.x_GLF_uniform_float_values[0].el);
      {
        i = (i + 1);
      }
      continue;
    }
  }
  bool x_60 = (f > x_6.x_GLF_uniform_float_values[1].el);
  x_67 = x_60;
  if (x_60) {
    x_66 = (f < x_6.x_GLF_uniform_float_values[2].el);
    x_67 = x_66;
  }
  if (x_67) {
    float v_1 = float(x_9.x_GLF_uniform_int_values[2].el);
    float v_2 = float(x_9.x_GLF_uniform_int_values[1].el);
    float v_3 = float(x_9.x_GLF_uniform_int_values[1].el);
    x_GLF_color = vec4(v_1, v_2, v_3, float(x_9.x_GLF_uniform_int_values[2].el));
  } else {
    x_GLF_color = vec4(float(x_9.x_GLF_uniform_int_values[1].el));
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
