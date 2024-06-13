fn countTrailingZeros_327c37() -> vec2<i32> {
  var res : vec2<i32> = countTrailingZeros(vec2<i32>(1i));
  return res;
}

@group(0) @binding(0) var<storage, read_write> prevent_dce : vec2<i32>;

@fragment
fn fragment_main() {
  prevent_dce = countTrailingZeros_327c37();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = countTrailingZeros_327c37();
}

struct VertexOutput {
  @builtin(position)
  pos : vec4<f32>,
  @location(0) @interpolate(flat)
  prevent_dce : vec2<i32>,
}

@vertex
fn vertex_main() -> VertexOutput {
  var out : VertexOutput;
  out.pos = vec4<f32>();
  out.prevent_dce = countTrailingZeros_327c37();
  return out;
}
