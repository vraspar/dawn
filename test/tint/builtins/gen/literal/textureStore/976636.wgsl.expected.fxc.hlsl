SKIP: FAILED


enable chromium_experimental_read_write_storage_texture;

@group(1) @binding(0) var arg_0 : texture_storage_1d<rgba8sint, read_write>;

fn textureStore_976636() {
  textureStore(arg_0, 1u, vec4<i32>(1i));
}

@vertex
fn vertex_main() -> @builtin(position) vec4<f32> {
  textureStore_976636();
  return vec4<f32>();
}

@fragment
fn fragment_main() {
  textureStore_976636();
}

@compute @workgroup_size(1)
fn compute_main() {
  textureStore_976636();
}

Failed to generate: builtins/gen/literal/textureStore/976636.wgsl:24:8 error: HLSL backend does not support extension 'chromium_experimental_read_write_storage_texture'
enable chromium_experimental_read_write_storage_texture;
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

