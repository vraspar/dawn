SKIP: FAILED

#version 310 es
precision highp float;
precision highp int;


struct VertexOutput {
  vec4 pos;
  vec2 prevent_dce;
};

vec2 prevent_dce;
vec2 sinh_b9860e() {
  vec2 res = vec2(1.17520117759704589844f);
  return res;
}
void main() {
  prevent_dce = sinh_b9860e();
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  prevent_dce = sinh_b9860e();
}
VertexOutput main() {
  VertexOutput tint_symbol = VertexOutput(vec4(0.0f), vec2(0.0f));
  tint_symbol.pos = vec4(0.0f);
  tint_symbol.prevent_dce = sinh_b9860e();
  return tint_symbol;
}
error: Error parsing GLSL shader:
ERROR: 0:19: 'local_size_x' : there is no such layout identifier for this stage taking an assigned value 
ERROR: 0:19: '' : compilation terminated 
ERROR: 2 compilation errors.  No code generated.



#version 310 es
precision highp float;
precision highp int;


struct VertexOutput {
  vec4 pos;
  vec2 prevent_dce;
};

vec2 prevent_dce;
vec2 sinh_b9860e() {
  vec2 res = vec2(1.17520117759704589844f);
  return res;
}
void main() {
  prevent_dce = sinh_b9860e();
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  prevent_dce = sinh_b9860e();
}
VertexOutput main() {
  VertexOutput tint_symbol = VertexOutput(vec4(0.0f), vec2(0.0f));
  tint_symbol.pos = vec4(0.0f);
  tint_symbol.prevent_dce = sinh_b9860e();
  return tint_symbol;
}
error: Error parsing GLSL shader:
ERROR: 0:20: 'main' : function already has a body 
ERROR: 0:20: '' : compilation terminated 
ERROR: 2 compilation errors.  No code generated.



#version 310 es
precision highp float;
precision highp int;


struct VertexOutput {
  vec4 pos;
  vec2 prevent_dce;
};

vec2 prevent_dce;
vec2 sinh_b9860e() {
  vec2 res = vec2(1.17520117759704589844f);
  return res;
}
void main() {
  prevent_dce = sinh_b9860e();
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  prevent_dce = sinh_b9860e();
}
VertexOutput main() {
  VertexOutput tint_symbol = VertexOutput(vec4(0.0f), vec2(0.0f));
  tint_symbol.pos = vec4(0.0f);
  tint_symbol.prevent_dce = sinh_b9860e();
  return tint_symbol;
}
error: Error parsing GLSL shader:
ERROR: 0:19: 'local_size_x' : there is no such layout identifier for this stage taking an assigned value 
ERROR: 0:19: '' : compilation terminated 
ERROR: 2 compilation errors.  No code generated.




tint executable returned error: exit status 1
