enable chromium_experimental_read_write_storage_texture;

@group(1) @binding(0) var arg_0 : texture_storage_2d_array<rgba8unorm, read_write>;

fn textureStore_d0fadc() {
  var arg_1 = vec2<i32>(1i);
  var arg_2 = 1u;
  var arg_3 = vec4<f32>(1.0f);
  textureStore(arg_0, arg_1, arg_2, arg_3);
}

@vertex
fn vertex_main() -> @builtin(position) vec4<f32> {
  textureStore_d0fadc();
  return vec4<f32>();
}

@fragment
fn fragment_main() {
  textureStore_d0fadc();
}

@compute @workgroup_size(1)
fn compute_main() {
  textureStore_d0fadc();
}
