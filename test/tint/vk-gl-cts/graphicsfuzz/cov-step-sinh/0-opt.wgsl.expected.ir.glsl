SKIP: FAILED

#version 310 es

struct main_out {
  vec4 x_GLF_color_1;
};
precision highp float;
precision highp int;


vec4 x_GLF_color = vec4(0.0f);
void main_1() {
  vec2 v1 = vec2(0.0f);
  vec2 v2 = vec2(0.0f);
  v1 = vec2(1.0f, -1.0f);
  vec2 x_22 = v1;
  v2 = step(vec2(0.40000000596046447754f), sinh(x_22));
  float x_27 = v2.x;
  float x_29 = v2.y;
  float x_31 = v2.y;
  float x_33 = v2.x;
  x_GLF_color = vec4(x_27, x_29, x_31, x_33);
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
