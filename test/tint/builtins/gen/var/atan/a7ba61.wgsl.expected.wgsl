enable f16;

fn atan_a7ba61() -> f16 {
  var arg_0 = 1.0h;
  var res : f16 = atan(arg_0);
  return res;
}

@group(0) @binding(0) var<storage, read_write> prevent_dce : f16;

@fragment
fn fragment_main() {
  prevent_dce = atan_a7ba61();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = atan_a7ba61();
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
  out.prevent_dce = atan_a7ba61();
  return out;
}
