var<workgroup> arg_0 : atomic<u32>;

fn atomicMax_beccfc() -> u32 {
  var res : u32 = atomicMax(&(arg_0), 1u);
  return res;
}

@group(0) @binding(0) var<storage, read_write> prevent_dce : u32;

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = atomicMax_beccfc();
}
