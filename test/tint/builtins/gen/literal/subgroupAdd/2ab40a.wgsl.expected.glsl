SKIP: INVALID


enable f16;

@group(0) @binding(0) var<storage, read_write> prevent_dce : vec4<f16>;

fn subgroupAdd_2ab40a() -> vec4<f16> {
  var res : vec4<f16> = subgroupAdd(vec4<f16>(1.0h));
  return res;
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = subgroupAdd_2ab40a();
}

Failed to generate: error: Unknown builtin method: 0x558870aff230
