SKIP: FAILED


enable chromium_experimental_read_write_storage_texture;

@group(1) @binding(0) var arg_0 : texture_storage_3d<rgba32uint, read_write>;

fn textureStore_d3a22b() {
  textureStore(arg_0, vec3<u32>(1u), vec4<u32>(1u));
}

@vertex
fn vertex_main() -> @builtin(position) vec4<f32> {
  textureStore_d3a22b();
  return vec4<f32>();
}

@fragment
fn fragment_main() {
  textureStore_d3a22b();
}

@compute @workgroup_size(1)
fn compute_main() {
  textureStore_d3a22b();
}

Failed to generate: builtins/gen/literal/textureStore/d3a22b.wgsl:24:8 error: HLSL backend does not support extension 'chromium_experimental_read_write_storage_texture'
enable chromium_experimental_read_write_storage_texture;
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

