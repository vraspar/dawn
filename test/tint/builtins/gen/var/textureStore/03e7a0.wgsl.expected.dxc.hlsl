SKIP: FAILED


enable chromium_experimental_read_write_storage_texture;

@group(1) @binding(0) var arg_0 : texture_storage_2d<rgba16uint, read_write>;

fn textureStore_03e7a0() {
  var arg_1 = vec2<i32>(1i);
  var arg_2 = vec4<u32>(1u);
  textureStore(arg_0, arg_1, arg_2);
}

@vertex
fn vertex_main() -> @builtin(position) vec4<f32> {
  textureStore_03e7a0();
  return vec4<f32>();
}

@fragment
fn fragment_main() {
  textureStore_03e7a0();
}

@compute @workgroup_size(1)
fn compute_main() {
  textureStore_03e7a0();
}

Failed to generate: builtins/gen/var/textureStore/03e7a0.wgsl:24:8 error: HLSL backend does not support extension 'chromium_experimental_read_write_storage_texture'
enable chromium_experimental_read_write_storage_texture;
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

