SKIP: INVALID


enable subgroups;

@group(0) @binding(0) var<storage, read_write> prevent_dce : i32;

fn quadSwapY_94ab6d() -> i32 {
  var arg_0 = 1i;
  var res : i32 = quadSwapY(arg_0);
  return res;
}

@fragment
fn fragment_main() {
  prevent_dce = quadSwapY_94ab6d();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = quadSwapY_94ab6d();
}

Failed to generate: <dawn>/test/tint/builtins/gen/var/quadSwapY/94ab6d.wgsl:41:8 error: GLSL backend does not support extension 'subgroups'
enable subgroups;
       ^^^^^^^^^


enable subgroups;

@group(0) @binding(0) var<storage, read_write> prevent_dce : i32;

fn quadSwapY_94ab6d() -> i32 {
  var arg_0 = 1i;
  var res : i32 = quadSwapY(arg_0);
  return res;
}

@fragment
fn fragment_main() {
  prevent_dce = quadSwapY_94ab6d();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = quadSwapY_94ab6d();
}

Failed to generate: <dawn>/test/tint/builtins/gen/var/quadSwapY/94ab6d.wgsl:41:8 error: GLSL backend does not support extension 'subgroups'
enable subgroups;
       ^^^^^^^^^

