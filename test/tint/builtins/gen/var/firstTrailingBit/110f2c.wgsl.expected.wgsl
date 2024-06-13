fn firstTrailingBit_110f2c() -> vec4<u32> {
  var arg_0 = vec4<u32>(1u);
  var res : vec4<u32> = firstTrailingBit(arg_0);
  return res;
}

@group(0) @binding(0) var<storage, read_write> prevent_dce : vec4<u32>;

@fragment
fn fragment_main() {
  prevent_dce = firstTrailingBit_110f2c();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = firstTrailingBit_110f2c();
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
  out.prevent_dce = firstTrailingBit_110f2c();
  return out;
}
