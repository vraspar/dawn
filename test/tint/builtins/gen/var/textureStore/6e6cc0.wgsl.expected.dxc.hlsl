SKIP: FAILED


enable chromium_experimental_read_write_storage_texture;

@group(1) @binding(0) var arg_0 : texture_storage_1d<r32float, read_write>;

fn textureStore_6e6cc0() {
  var arg_1 = 1i;
  var arg_2 = vec4<f32>(1.0f);
  textureStore(arg_0, arg_1, arg_2);
}

@vertex
fn vertex_main() -> @builtin(position) vec4<f32> {
  textureStore_6e6cc0();
  return vec4<f32>();
}

@fragment
fn fragment_main() {
  textureStore_6e6cc0();
}

@compute @workgroup_size(1)
fn compute_main() {
  textureStore_6e6cc0();
}

Failed to generate: builtins/gen/var/textureStore/6e6cc0.wgsl:24:8 error: HLSL backend does not support extension 'chromium_experimental_read_write_storage_texture'
enable chromium_experimental_read_write_storage_texture;
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

