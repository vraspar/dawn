SKIP: FAILED

#version 310 es

struct main_out {
  vec4 x_GLF_color_1;
};
precision highp float;
precision highp int;


vec4 tint_symbol = vec4(0.0f);
vec4 x_GLF_color = vec4(0.0f);
float func_() {
  float x = 0.0f;
  x = 2.0f;
  float x_35 = tint_symbol.x;
  if ((x_35 == 12.0f)) {
    float x_40 = tint_symbol.y;
    if ((x_40 == 13.0f)) {
      float x_44 = x;
      x = (x_44 + 1.0f);
    }
    float x_46 = x;
    return x_46;
  }
  return 1.0f;
}
void main_1() {
  if (false) {
    float x_31 = func_();
    x_GLF_color = vec4(x_31, x_31, x_31, x_31);
  } else {
    x_GLF_color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
  }
}
main_out main(vec4 tint_symbol_2) {
  tint_symbol = tint_symbol_2;
  main_1();
  return main_out(x_GLF_color);
}
error: Error parsing GLSL shader:
ERROR: 0:4: 'float' : type requires declaration of default precision qualifier 
ERROR: 0:4: '' : compilation terminated 
ERROR: 2 compilation errors.  No code generated.




tint executable returned error: exit status 1
