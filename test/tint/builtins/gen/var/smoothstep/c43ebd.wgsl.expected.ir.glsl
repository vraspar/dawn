SKIP: FAILED

#version 310 es
#extension GL_AMD_gpu_shader_half_float: require
precision highp float;
precision highp int;


struct VertexOutput {
  vec4 pos;
  f16vec4 prevent_dce;
};

f16vec4 prevent_dce;
f16vec4 smoothstep_c43ebd() {
  f16vec4 arg_0 = f16vec4(2.0hf);
  f16vec4 arg_1 = f16vec4(4.0hf);
  f16vec4 arg_2 = f16vec4(3.0hf);
  f16vec4 res = smoothstep(arg_0, arg_1, arg_2);
  return res;
}
void main() {
  prevent_dce = smoothstep_c43ebd();
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  prevent_dce = smoothstep_c43ebd();
}
VertexOutput main() {
  VertexOutput tint_symbol = VertexOutput(vec4(0.0f), f16vec4(0.0hf));
  tint_symbol.pos = vec4(0.0f);
  tint_symbol.prevent_dce = smoothstep_c43ebd();
  return tint_symbol;
}
error: Error parsing GLSL shader:
ERROR: 0:23: 'local_size_x' : there is no such layout identifier for this stage taking an assigned value 
ERROR: 0:23: '' : compilation terminated 
ERROR: 2 compilation errors.  No code generated.



#version 310 es
#extension GL_AMD_gpu_shader_half_float: require
precision highp float;
precision highp int;


struct VertexOutput {
  vec4 pos;
  f16vec4 prevent_dce;
};

f16vec4 prevent_dce;
f16vec4 smoothstep_c43ebd() {
  f16vec4 arg_0 = f16vec4(2.0hf);
  f16vec4 arg_1 = f16vec4(4.0hf);
  f16vec4 arg_2 = f16vec4(3.0hf);
  f16vec4 res = smoothstep(arg_0, arg_1, arg_2);
  return res;
}
void main() {
  prevent_dce = smoothstep_c43ebd();
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  prevent_dce = smoothstep_c43ebd();
}
VertexOutput main() {
  VertexOutput tint_symbol = VertexOutput(vec4(0.0f), f16vec4(0.0hf));
  tint_symbol.pos = vec4(0.0f);
  tint_symbol.prevent_dce = smoothstep_c43ebd();
  return tint_symbol;
}
error: Error parsing GLSL shader:
ERROR: 0:24: 'main' : function already has a body 
ERROR: 0:24: '' : compilation terminated 
ERROR: 2 compilation errors.  No code generated.



#version 310 es
#extension GL_AMD_gpu_shader_half_float: require
precision highp float;
precision highp int;


struct VertexOutput {
  vec4 pos;
  f16vec4 prevent_dce;
};

f16vec4 prevent_dce;
f16vec4 smoothstep_c43ebd() {
  f16vec4 arg_0 = f16vec4(2.0hf);
  f16vec4 arg_1 = f16vec4(4.0hf);
  f16vec4 arg_2 = f16vec4(3.0hf);
  f16vec4 res = smoothstep(arg_0, arg_1, arg_2);
  return res;
}
void main() {
  prevent_dce = smoothstep_c43ebd();
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  prevent_dce = smoothstep_c43ebd();
}
VertexOutput main() {
  VertexOutput tint_symbol = VertexOutput(vec4(0.0f), f16vec4(0.0hf));
  tint_symbol.pos = vec4(0.0f);
  tint_symbol.prevent_dce = smoothstep_c43ebd();
  return tint_symbol;
}
error: Error parsing GLSL shader:
ERROR: 0:23: 'local_size_x' : there is no such layout identifier for this stage taking an assigned value 
ERROR: 0:23: '' : compilation terminated 
ERROR: 2 compilation errors.  No code generated.




tint executable returned error: exit status 1
