@group(1) @binding(0) var arg_0 : texture_depth_2d_array;

fn textureLoad_ff1119() -> f32 {
  var res : f32 = textureLoad(arg_0, vec2<i32>(1i), 1i, 1u);
  return res;
}

@group(0) @binding(0) var<storage, read_write> prevent_dce : f32;

@fragment
fn fragment_main() {
  prevent_dce = textureLoad_ff1119();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = textureLoad_ff1119();
}

struct VertexOutput {
  @builtin(position)
  pos : vec4<f32>,
  @location(0) @interpolate(flat)
  prevent_dce : f32,
}

@vertex
fn vertex_main() -> VertexOutput {
  var out : VertexOutput;
  out.pos = vec4<f32>();
  out.prevent_dce = textureLoad_ff1119();
  return out;
}
