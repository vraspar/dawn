SKIP: FAILED

#version 310 es
precision highp float;
precision highp int;


struct VertexOutput {
  vec4 pos;
  uvec4 prevent_dce;
};

uvec4 prevent_dce;
uvec4 clamp_bd43ce() {
  uvec4 arg_0 = uvec4(1u);
  uvec4 arg_1 = uvec4(1u);
  uvec4 arg_2 = uvec4(1u);
  uvec4 v = arg_2;
  uvec4 res = min(max(arg_0, arg_1), v);
  return res;
}
void main() {
  prevent_dce = clamp_bd43ce();
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  prevent_dce = clamp_bd43ce();
}
VertexOutput main() {
  VertexOutput tint_symbol = VertexOutput(vec4(0.0f), uvec4(0u));
  tint_symbol.pos = vec4(0.0f);
  tint_symbol.prevent_dce = clamp_bd43ce();
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
  uvec4 prevent_dce;
};

uvec4 prevent_dce;
uvec4 clamp_bd43ce() {
  uvec4 arg_0 = uvec4(1u);
  uvec4 arg_1 = uvec4(1u);
  uvec4 arg_2 = uvec4(1u);
  uvec4 v = arg_2;
  uvec4 res = min(max(arg_0, arg_1), v);
  return res;
}
void main() {
  prevent_dce = clamp_bd43ce();
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  prevent_dce = clamp_bd43ce();
}
VertexOutput main() {
  VertexOutput tint_symbol = VertexOutput(vec4(0.0f), uvec4(0u));
  tint_symbol.pos = vec4(0.0f);
  tint_symbol.prevent_dce = clamp_bd43ce();
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
  uvec4 prevent_dce;
};

uvec4 prevent_dce;
uvec4 clamp_bd43ce() {
  uvec4 arg_0 = uvec4(1u);
  uvec4 arg_1 = uvec4(1u);
  uvec4 arg_2 = uvec4(1u);
  uvec4 v = arg_2;
  uvec4 res = min(max(arg_0, arg_1), v);
  return res;
}
void main() {
  prevent_dce = clamp_bd43ce();
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  prevent_dce = clamp_bd43ce();
}
VertexOutput main() {
  VertexOutput tint_symbol = VertexOutput(vec4(0.0f), uvec4(0u));
  tint_symbol.pos = vec4(0.0f);
  tint_symbol.prevent_dce = clamp_bd43ce();
  return tint_symbol;
}
error: Error parsing GLSL shader:
ERROR: 0:23: 'local_size_x' : there is no such layout identifier for this stage taking an assigned value 
ERROR: 0:23: '' : compilation terminated 
ERROR: 2 compilation errors.  No code generated.




tint executable returned error: exit status 1
