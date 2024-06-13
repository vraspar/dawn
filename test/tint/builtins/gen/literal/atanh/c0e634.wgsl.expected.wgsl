fn atanh_c0e634() -> vec2<f32> {
  var res : vec2<f32> = atanh(vec2<f32>(0.5f));
  return res;
}

@group(0) @binding(0) var<storage, read_write> prevent_dce : vec2<f32>;

@fragment
fn fragment_main() {
  prevent_dce = atanh_c0e634();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = atanh_c0e634();
}

struct VertexOutput {
  @builtin(position)
  pos : vec4<f32>,
  @location(0) @interpolate(flat)
  prevent_dce : vec2<f32>,
}

@vertex
fn vertex_main() -> VertexOutput {
  var out : VertexOutput;
  out.pos = vec4<f32>();
  out.prevent_dce = atanh_c0e634();
  return out;
}
