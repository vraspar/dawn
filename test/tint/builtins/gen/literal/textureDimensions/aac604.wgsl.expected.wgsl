@group(1) @binding(0) var arg_0 : texture_1d<f32>;

fn textureDimensions_aac604() -> u32 {
  var res : u32 = textureDimensions(arg_0, 1u);
  return res;
}

@group(0) @binding(0) var<storage, read_write> prevent_dce : u32;

@fragment
fn fragment_main() {
  prevent_dce = textureDimensions_aac604();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = textureDimensions_aac604();
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
  out.prevent_dce = textureDimensions_aac604();
  return out;
}
