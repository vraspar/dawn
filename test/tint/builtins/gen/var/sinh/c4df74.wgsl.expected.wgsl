fn sinh_c4df74() {
  const arg_0 = 1.0;
  var res = sinh(arg_0);
}

@fragment
fn fragment_main() {
  sinh_c4df74();
}

@compute @workgroup_size(1)
fn compute_main() {
  sinh_c4df74();
}

struct VertexOutput {
  @builtin(position)
  pos : vec4<f32>,
}

@vertex
fn vertex_main() -> VertexOutput {
  var out : VertexOutput;
  out.pos = vec4<f32>();
  sinh_c4df74();
  return out;
}
