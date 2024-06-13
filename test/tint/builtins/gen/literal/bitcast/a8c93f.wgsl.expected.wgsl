fn bitcast_a8c93f() -> vec4<u32> {
  var res : vec4<u32> = bitcast<vec4<u32>>(vec4<i32>(1i));
  return res;
}

@group(0) @binding(0) var<storage, read_write> prevent_dce : vec4<u32>;

@fragment
fn fragment_main() {
  prevent_dce = bitcast_a8c93f();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = bitcast_a8c93f();
}

struct VertexOutput {
  @builtin(position)
  pos : vec4<f32>,
  @location(0) @interpolate(flat)
  prevent_dce : vec4<u32>,
}

@vertex
fn vertex_main() -> VertexOutput {
  var out : VertexOutput;
  out.pos = vec4<f32>();
  out.prevent_dce = bitcast_a8c93f();
  return out;
}
