enable subgroups;

@group(0) @binding(0) var<storage, read_write> prevent_dce : vec2<f32>;

fn subgroupMax_1fc846() -> vec2<f32> {
  var res : vec2<f32> = subgroupMax(vec2<f32>(1.0f));
  return res;
}

@fragment
fn fragment_main() {
  prevent_dce = subgroupMax_1fc846();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = subgroupMax_1fc846();
}
