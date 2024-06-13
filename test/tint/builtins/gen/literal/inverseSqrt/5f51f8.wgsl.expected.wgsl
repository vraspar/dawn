enable f16;

fn inverseSqrt_5f51f8() -> vec2<f16> {
  var res : vec2<f16> = inverseSqrt(vec2<f16>(1.0h));
  return res;
}

@group(0) @binding(0) var<storage, read_write> prevent_dce : vec2<f16>;

@fragment
fn fragment_main() {
  prevent_dce = inverseSqrt_5f51f8();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = inverseSqrt_5f51f8();
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
  out.prevent_dce = inverseSqrt_5f51f8();
  return out;
}
