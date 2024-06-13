enable f16;

fn sqrt_d9ab4d() -> vec2<f16> {
  var res : vec2<f16> = sqrt(vec2<f16>(1.0h));
  return res;
}

@group(0) @binding(0) var<storage, read_write> prevent_dce : vec2<f16>;

@fragment
fn fragment_main() {
  prevent_dce = sqrt_d9ab4d();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = sqrt_d9ab4d();
}

struct VertexOutput {
  @builtin(position)
  pos : vec4<f32>,
  @location(0) @interpolate(flat)
  prevent_dce : vec2<f16>,
}

@vertex
fn vertex_main() -> VertexOutput {
  var out : VertexOutput;
  out.pos = vec4<f32>();
  out.prevent_dce = sqrt_d9ab4d();
  return out;
}
