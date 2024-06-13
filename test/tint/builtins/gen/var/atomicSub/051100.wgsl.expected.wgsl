struct SB_RW {
  arg_0 : atomic<i32>,
}

@group(0) @binding(1) var<storage, read_write> sb_rw : SB_RW;

fn atomicSub_051100() -> i32 {
  var arg_1 = 1i;
  var res : i32 = atomicSub(&(sb_rw.arg_0), arg_1);
  return res;
}

@group(0) @binding(0) var<storage, read_write> prevent_dce : i32;

@fragment
fn fragment_main() {
  prevent_dce = atomicSub_051100();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = atomicSub_051100();
}
