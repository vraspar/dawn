SKIP: FAILED

#version 310 es

struct main_out {
  vec4 tint_symbol;
};

vec4 position_1 = vec4(0.0f);
vec4 tint_symbol = vec4(0.0f);
void main_1() {
  vec2 x_23 = position_1.xy;
  tint_symbol = vec4(x_23[0u], x_23[1u], 0.69999998807907104492f, 1.0f);
}
main_out main(vec4 position_1_param) {
  position_1 = position_1_param;
  main_1();
  return main_out(tint_symbol);
}
error: Error parsing GLSL shader:
ERROR: 0:13: 'main' : function cannot take any parameter(s) 
ERROR: 0:13: 'structure' :  entry point cannot return a value
ERROR: 0:13: '' : compilation terminated 
ERROR: 3 compilation errors.  No code generated.




tint executable returned error: exit status 1
