SKIP: FAILED

#version 310 es

struct main_out {
  vec4 x_GLF_color_1;
};
precision highp float;
precision highp int;


vec4 x_GLF_color = vec4(0.0f);
void main_1() {
  bool c1 = false;
  vec2 uv = vec2(0.0f);
  int i = 0;
  bool x_37 = false;
  int x_9 = 0;
  bool x_35 = (uv.y < 0.25f);
  c1 = x_35;
  i = 0;
  x_37 = x_35;
  x_9 = 0;
  {
    while(true) {
      if ((x_9 < 1)) {
      } else {
        break;
      }
      x_GLF_color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
      return;
    }
  }
  if (x_37) {
    return;
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
