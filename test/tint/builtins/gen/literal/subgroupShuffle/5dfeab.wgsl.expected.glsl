SKIP: INVALID


enable subgroups;

@group(0) @binding(0) var<storage, read_write> prevent_dce : vec4<f32>;

fn subgroupShuffle_5dfeab() -> vec4<f32> {
  var res : vec4<f32> = subgroupShuffle(vec4<f32>(1.0f), 1i);
  return res;
}

@fragment
fn fragment_main() {
  prevent_dce = subgroupShuffle_5dfeab();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = subgroupShuffle_5dfeab();
}

Failed to generate: <dawn>/test/tint/builtins/gen/literal/subgroupShuffle/5dfeab.wgsl:41:8 error: GLSL backend does not support extension 'subgroups'
enable subgroups;
       ^^^^^^^^^


enable subgroups;

@group(0) @binding(0) var<storage, read_write> prevent_dce : vec4<f32>;

fn subgroupShuffle_5dfeab() -> vec4<f32> {
  var res : vec4<f32> = subgroupShuffle(vec4<f32>(1.0f), 1i);
  return res;
}

@fragment
fn fragment_main() {
  prevent_dce = subgroupShuffle_5dfeab();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = subgroupShuffle_5dfeab();
}

Failed to generate: <dawn>/test/tint/builtins/gen/literal/subgroupShuffle/5dfeab.wgsl:41:8 error: GLSL backend does not support extension 'subgroups'
enable subgroups;
       ^^^^^^^^^

