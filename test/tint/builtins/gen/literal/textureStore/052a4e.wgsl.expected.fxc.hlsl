SKIP: FAILED


enable chromium_experimental_read_write_storage_texture;

@group(1) @binding(0) var arg_0 : texture_storage_1d<rgba8unorm, read_write>;

fn textureStore_052a4e() {
  textureStore(arg_0, 1u, vec4<f32>(1.0f));
}

@vertex
fn vertex_main() -> @builtin(position) vec4<f32> {
  textureStore_052a4e();
  return vec4<f32>();
}

@fragment
fn fragment_main() {
  textureStore_052a4e();
}

@compute @workgroup_size(1)
fn compute_main() {
  textureStore_052a4e();
}

Failed to generate: builtins/gen/literal/textureStore/052a4e.wgsl:24:8 error: HLSL backend does not support extension 'chromium_experimental_read_write_storage_texture'
enable chromium_experimental_read_write_storage_texture;
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

