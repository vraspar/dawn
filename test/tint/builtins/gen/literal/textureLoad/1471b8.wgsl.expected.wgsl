@group(1) @binding(0) var arg_0 : texture_storage_2d_array<rgba32sint, read>;

fn textureLoad_1471b8() -> vec4<i32> {
  var res : vec4<i32> = textureLoad(arg_0, vec2<u32>(1u), 1i);
  return res;
}

@group(0) @binding(0) var<storage, read_write> prevent_dce : vec4<i32>;

@fragment
fn fragment_main() {
  prevent_dce = textureLoad_1471b8();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = textureLoad_1471b8();
}

struct VertexOutput {
  @builtin(position)
  pos : vec4<f32>,
  @location(0) @interpolate(flat)
  prevent_dce : vec4<i32>,
}

@vertex
fn vertex_main() -> VertexOutput {
  var out : VertexOutput;
  out.pos = vec4<f32>();
  out.prevent_dce = textureLoad_1471b8();
  return out;
}
