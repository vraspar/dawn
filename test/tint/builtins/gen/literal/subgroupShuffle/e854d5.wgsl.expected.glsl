SKIP: INVALID


enable subgroups;

@group(0) @binding(0) var<storage, read_write> prevent_dce : vec4<i32>;

fn subgroupShuffle_e854d5() -> vec4<i32> {
  var res : vec4<i32> = subgroupShuffle(vec4<i32>(1i), 1u);
  return res;
}

@fragment
fn fragment_main() {
  prevent_dce = subgroupShuffle_e854d5();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = subgroupShuffle_e854d5();
}

Failed to generate: <dawn>/test/tint/builtins/gen/literal/subgroupShuffle/e854d5.wgsl:41:8 error: GLSL backend does not support extension 'subgroups'
enable subgroups;
       ^^^^^^^^^


enable subgroups;

@group(0) @binding(0) var<storage, read_write> prevent_dce : vec4<i32>;

fn subgroupShuffle_e854d5() -> vec4<i32> {
  var res : vec4<i32> = subgroupShuffle(vec4<i32>(1i), 1u);
  return res;
}

@fragment
fn fragment_main() {
  prevent_dce = subgroupShuffle_e854d5();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = subgroupShuffle_e854d5();
}

Failed to generate: <dawn>/test/tint/builtins/gen/literal/subgroupShuffle/e854d5.wgsl:41:8 error: GLSL backend does not support extension 'subgroups'
enable subgroups;
       ^^^^^^^^^

