SKIP: INVALID


enable subgroups;

@group(0) @binding(0) var<storage, read_write> prevent_dce : vec4<f32>;

fn quadSwapY_b9d9e7() -> vec4<f32> {
  var res : vec4<f32> = quadSwapY(vec4<f32>(1.0f));
  return res;
}

@fragment
fn fragment_main() {
  prevent_dce = quadSwapY_b9d9e7();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = quadSwapY_b9d9e7();
}

Failed to generate: <dawn>/test/tint/builtins/gen/literal/quadSwapY/b9d9e7.wgsl:41:8 error: GLSL backend does not support extension 'subgroups'
enable subgroups;
       ^^^^^^^^^


enable subgroups;

@group(0) @binding(0) var<storage, read_write> prevent_dce : vec4<f32>;

fn quadSwapY_b9d9e7() -> vec4<f32> {
  var res : vec4<f32> = quadSwapY(vec4<f32>(1.0f));
  return res;
}

@fragment
fn fragment_main() {
  prevent_dce = quadSwapY_b9d9e7();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = quadSwapY_b9d9e7();
}

Failed to generate: <dawn>/test/tint/builtins/gen/literal/quadSwapY/b9d9e7.wgsl:41:8 error: GLSL backend does not support extension 'subgroups'
enable subgroups;
       ^^^^^^^^^

