SKIP: FAILED


enable chromium_experimental_read_write_storage_texture;

@group(1) @binding(0) var arg_0 : texture_storage_2d<rgba16sint, read_write>;

fn textureStore_4ddf52() {
  textureStore(arg_0, vec2<u32>(1u), vec4<i32>(1i));
}

@vertex
fn vertex_main() -> @builtin(position) vec4<f32> {
  textureStore_4ddf52();
  return vec4<f32>();
}

@fragment
fn fragment_main() {
  textureStore_4ddf52();
}

@compute @workgroup_size(1)
fn compute_main() {
  textureStore_4ddf52();
}

Failed to generate: builtins/gen/literal/textureStore/4ddf52.wgsl:24:8 error: HLSL backend does not support extension 'chromium_experimental_read_write_storage_texture'
enable chromium_experimental_read_write_storage_texture;
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

