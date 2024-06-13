@group(1) @binding(1) var arg_1 : texture_2d<u32>;

@group(1) @binding(2) var arg_2 : sampler;

fn textureGather_8fae00() -> vec4<u32> {
  var res : vec4<u32> = textureGather(1u, arg_1, arg_2, vec2<f32>(1.0f));
  return res;
}

@group(0) @binding(0) var<storage, read_write> prevent_dce : vec4<u32>;

@fragment
fn fragment_main() {
  prevent_dce = textureGather_8fae00();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = textureGather_8fae00();
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
  out.prevent_dce = textureGather_8fae00();
  return out;
}
