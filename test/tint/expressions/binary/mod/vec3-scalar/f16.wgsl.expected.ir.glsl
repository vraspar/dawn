SKIP: FAILED

#version 310 es
#extension GL_AMD_gpu_shader_half_float: require

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  f16vec3 a = f16vec3(1.0hf, 2.0hf, 3.0hf);
  float16_t b = 4.0hf;
  f16vec3 r = (a % b);
}
error: Error parsing GLSL shader:
ERROR: 0:8: '%' :  wrong operand types: no operation '%' exists that takes a left-hand operand of type ' temp 3-component vector of float16_t' and a right operand of type ' temp float16_t' (or there is no acceptable conversion)
ERROR: 0:8: '' : compilation terminated 
ERROR: 2 compilation errors.  No code generated.




tint executable returned error: exit status 1
