SKIP: FAILED


enable chromium_experimental_read_write_storage_texture;

@group(1) @binding(0) var arg_0 : texture_storage_2d_array<rg32float, read_write>;

fn textureStore_4e2b3a() {
  textureStore(arg_0, vec2<i32>(1i), 1i, vec4<f32>(1.0f));
}

@vertex
fn vertex_main() -> @builtin(position) vec4<f32> {
  textureStore_4e2b3a();
  return vec4<f32>();
}

@fragment
fn fragment_main() {
  textureStore_4e2b3a();
}

@compute @workgroup_size(1)
fn compute_main() {
  textureStore_4e2b3a();
}

Failed to generate: builtins/gen/literal/textureStore/4e2b3a.wgsl:24:8 error: HLSL backend does not support extension 'chromium_experimental_read_write_storage_texture'
enable chromium_experimental_read_write_storage_texture;
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

