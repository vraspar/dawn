SKIP: FAILED

#version 310 es

struct main_out {
  vec4 x_GLF_color_1;
};
precision highp float;
precision highp int;


vec4 x_GLF_color = vec4(0.0f);
void main_1() {
  bool b = false;
  int i = 0;
  float a = 0.0f;
  b = false;
  i = 1;
  {
    while(true) {
      if ((i > 0)) {
      } else {
        break;
      }
      a = (3.0f - float(i));
      if (((2.0f - a) == 0.0f)) {
        b = true;
      }
      {
        i = (i - 1);
      }
      continue;
    }
  }
  if (b) {
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
