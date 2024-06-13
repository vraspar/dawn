@group(1) @binding(0) var arg_0 : texture_2d<f32>;

fn textureNumLevels_24b2c6() -> u32 {
  var res : u32 = textureNumLevels(arg_0);
  return res;
}

@group(0) @binding(0) var<storage, read_write> prevent_dce : u32;

@fragment
fn fragment_main() {
  prevent_dce = textureNumLevels_24b2c6();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = textureNumLevels_24b2c6();
}

struct VertexOutput {
  @builtin(position)
  pos : vec4<f32>,
  @location(0) @interpolate(flat)
  prevent_dce : u32,
}

@vertex
fn vertex_main() -> VertexOutput {
  var out : VertexOutput;
  out.pos = vec4<f32>();
  out.prevent_dce = textureNumLevels_24b2c6();
  return out;
}
