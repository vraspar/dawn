SKIP: FAILED

#version 310 es
precision highp float;
precision highp int;


struct FragmentInputs {
  int loc0;
  uint loc1;
  float loc2;
  vec4 loc3;
};

void main(FragmentInputs inputs) {
  int i = inputs.loc0;
  uint u = inputs.loc1;
  float f = inputs.loc2;
  vec4 v = inputs.loc3;
}
error: Error parsing GLSL shader:
ERROR: 0:13: 'main' : function cannot take any parameter(s) 
ERROR: 0:13: '' : compilation terminated 
ERROR: 2 compilation errors.  No code generated.




tint executable returned error: exit status 1
