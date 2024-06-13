#version 310 es

shared int arg_0;
void tint_zero_workgroup_memory(uint local_idx) {
  if ((local_idx < 1u)) {
    atomicExchange(arg_0, 0);
  }
  barrier();
}

int atomicXor_75dc95() {
  int arg_1 = 1;
  int res = atomicXor(arg_0, arg_1);
  return res;
}

layout(binding = 0, std430) buffer prevent_dce_block_ssbo {
  int inner;
} prevent_dce;

void compute_main(uint local_invocation_index) {
  tint_zero_workgroup_memory(local_invocation_index);
  prevent_dce.inner = atomicXor_75dc95();
}

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  compute_main(gl_LocalInvocationIndex);
  return;
}
