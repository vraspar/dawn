fn dpdx_99edb1() -> vec2<f32> {
  var res : vec2<f32> = dpdx(vec2<f32>(1.0f));
  return res;
}

@group(0) @binding(0) var<storage, read_write> prevent_dce : vec2<f32>;

@fragment
fn fragment_main() {
  prevent_dce = dpdx_99edb1();
}
