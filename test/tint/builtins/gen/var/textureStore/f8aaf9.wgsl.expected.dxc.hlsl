SKIP: FAILED


enable chromium_experimental_read_write_storage_texture;

@group(1) @binding(0) var arg_0 : texture_storage_2d_array<rgba32sint, read_write>;

fn textureStore_f8aaf9() {
  var arg_1 = vec2<i32>(1i);
  var arg_2 = 1i;
  var arg_3 = vec4<i32>(1i);
  textureStore(arg_0, arg_1, arg_2, arg_3);
}

@vertex
fn vertex_main() -> @builtin(position) vec4<f32> {
  textureStore_f8aaf9();
  return vec4<f32>();
}

@fragment
fn fragment_main() {
  textureStore_f8aaf9();
}

@compute @workgroup_size(1)
fn compute_main() {
  textureStore_f8aaf9();
}

Failed to generate: builtins/gen/var/textureStore/f8aaf9.wgsl:24:8 error: HLSL backend does not support extension 'chromium_experimental_read_write_storage_texture'
enable chromium_experimental_read_write_storage_texture;
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

