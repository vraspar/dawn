enable f16;

fn asin_11dfda() -> f16 {
  var res : f16 = asin(0.479248046875h);
  return res;
}

@group(0) @binding(0) var<storage, read_write> prevent_dce : f16;

@fragment
fn fragment_main() {
  prevent_dce = asin_11dfda();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = asin_11dfda();
}

struct VertexOutput {
  @builtin(position)
  pos : vec4<f32>,
  @location(0) @interpolate(flat)
  prevent_dce : f16,
}

@vertex
fn vertex_main() -> VertexOutput {
  var out : VertexOutput;
  out.pos = vec4<f32>();
  out.prevent_dce = asin_11dfda();
  return out;
}
