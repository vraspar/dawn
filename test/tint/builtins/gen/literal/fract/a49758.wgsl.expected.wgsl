fn fract_a49758() -> vec3<f32> {
  var res : vec3<f32> = fract(vec3<f32>(1.25f));
  return res;
}

@group(0) @binding(0) var<storage, read_write> prevent_dce : vec3<f32>;

@fragment
fn fragment_main() {
  prevent_dce = fract_a49758();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = fract_a49758();
}

struct VertexOutput {
  @builtin(position)
  pos : vec4<f32>,
  @location(0) @interpolate(flat)
  prevent_dce : vec3<f32>,
}

@vertex
fn vertex_main() -> VertexOutput {
  var out : VertexOutput;
  out.pos = vec4<f32>();
  out.prevent_dce = fract_a49758();
  return out;
}
