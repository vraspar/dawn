SKIP: FAILED

#version 310 es

struct strided_arr {
  int el;
};

struct buf0 {
  strided_arr x_GLF_uniform_int_values[2];
};

struct main_out {
  vec4 x_GLF_color_1;
};
precision highp float;
precision highp int;


int x_GLF_global_loop_count = 0;
vec4 x_GLF_color = vec4(0.0f);
uniform buf0 x_6;
void main_1() {
  x_GLF_global_loop_count = 0;
  {
    while(true) {
      if ((x_GLF_global_loop_count < 100)) {
      } else {
        break;
      }
      x_GLF_global_loop_count = (x_GLF_global_loop_count + 1);
      if (((x_GLF_global_loop_count * x_GLF_global_loop_count) > 10)) {
        break;
      }
      {
      }
      continue;
    }
  }
  if ((x_GLF_global_loop_count == 4)) {
    float v = float(x_6.x_GLF_uniform_int_values[0].el);
    float v_1 = float(x_6.x_GLF_uniform_int_values[1].el);
    float v_2 = float(x_6.x_GLF_uniform_int_values[1].el);
    x_GLF_color = vec4(v, v_1, v_2, float(x_6.x_GLF_uniform_int_values[0].el));
  } else {
    x_GLF_color = vec4(float(x_6.x_GLF_uniform_int_values[1].el));
  }
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
