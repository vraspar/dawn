SKIP: FAILED


enable chromium_experimental_read_write_storage_texture;

@group(1) @binding(0) var arg_0 : texture_storage_2d<bgra8unorm, read_write>;

fn textureStore_e46fd8() {
  textureStore(arg_0, vec2<u32>(1u), vec4<f32>(1.0f));
}

@vertex
fn vertex_main() -> @builtin(position) vec4<f32> {
  textureStore_e46fd8();
  return vec4<f32>();
}

@fragment
fn fragment_main() {
  textureStore_e46fd8();
}

@compute @workgroup_size(1)
fn compute_main() {
  textureStore_e46fd8();
}

Failed to generate: builtins/gen/literal/textureStore/e46fd8.wgsl:24:8 error: HLSL backend does not support extension 'chromium_experimental_read_write_storage_texture'
enable chromium_experimental_read_write_storage_texture;
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

