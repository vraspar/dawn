SKIP: FAILED

#version 310 es
precision highp float;
precision highp int;


struct VertexOutput {
  vec4 pos;
  ivec2 prevent_dce;
};

ivec2 prevent_dce;
ivec2 clamp_6c1749() {
  ivec2 arg_0 = ivec2(1);
  ivec2 arg_1 = ivec2(1);
  ivec2 arg_2 = ivec2(1);
  ivec2 v = arg_2;
  ivec2 res = min(max(arg_0, arg_1), v);
  return res;
}
void main() {
  prevent_dce = clamp_6c1749();
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  prevent_dce = clamp_6c1749();
}
VertexOutput main() {
  VertexOutput tint_symbol = VertexOutput(vec4(0.0f), ivec2(0));
  tint_symbol.pos = vec4(0.0f);
  tint_symbol.prevent_dce = clamp_6c1749();
  return tint_symbol;
}
error: Error parsing GLSL shader:
ERROR: 0:23: 'local_size_x' : there is no such layout identifier for this stage taking an assigned value 
ERROR: 0:23: '' : compilation terminated 
ERROR: 2 compilation errors.  No code generated.



#version 310 es
precision highp float;
precision highp int;


struct VertexOutput {
  vec4 pos;
  ivec2 prevent_dce;
};

ivec2 prevent_dce;
ivec2 clamp_6c1749() {
  ivec2 arg_0 = ivec2(1);
  ivec2 arg_1 = ivec2(1);
  ivec2 arg_2 = ivec2(1);
  ivec2 v = arg_2;
  ivec2 res = min(max(arg_0, arg_1), v);
  return res;
}
void main() {
  prevent_dce = clamp_6c1749();
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  prevent_dce = clamp_6c1749();
}
VertexOutput main() {
  VertexOutput tint_symbol = VertexOutput(vec4(0.0f), ivec2(0));
  tint_symbol.pos = vec4(0.0f);
  tint_symbol.prevent_dce = clamp_6c1749();
  return tint_symbol;
}
error: Error parsing GLSL shader:
ERROR: 0:24: 'main' : function already has a body 
ERROR: 0:24: '' : compilation terminated 
ERROR: 2 compilation errors.  No code generated.



#version 310 es
precision highp float;
precision highp int;


struct VertexOutput {
  vec4 pos;
  ivec2 prevent_dce;
};

ivec2 prevent_dce;
ivec2 clamp_6c1749() {
  ivec2 arg_0 = ivec2(1);
  ivec2 arg_1 = ivec2(1);
  ivec2 arg_2 = ivec2(1);
  ivec2 v = arg_2;
  ivec2 res = min(max(arg_0, arg_1), v);
  return res;
}
void main() {
  prevent_dce = clamp_6c1749();
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  prevent_dce = clamp_6c1749();
}
VertexOutput main() {
  VertexOutput tint_symbol = VertexOutput(vec4(0.0f), ivec2(0));
  tint_symbol.pos = vec4(0.0f);
  tint_symbol.prevent_dce = clamp_6c1749();
  return tint_symbol;
}
error: Error parsing GLSL shader:
ERROR: 0:23: 'local_size_x' : there is no such layout identifier for this stage taking an assigned value 
ERROR: 0:23: '' : compilation terminated 
ERROR: 2 compilation errors.  No code generated.




tint executable returned error: exit status 1
