#version 310 es
#extension GL_AMD_gpu_shader_half_float: require

f16mat4x2 m = f16mat4x2(f16vec2(0.0hf), f16vec2(0.0hf), f16vec2(0.0hf), f16vec2(0.0hf));
f16mat4x2 tint_symbol;
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  tint_symbol = m;
}
