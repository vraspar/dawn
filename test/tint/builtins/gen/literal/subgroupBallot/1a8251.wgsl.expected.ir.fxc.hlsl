SKIP: FAILED


RWByteAddressBuffer prevent_dce : register(u0);
uint4 subgroupBallot_1a8251() {
  uint4 res = WaveActiveBallot(true);
  return res;
}

[numthreads(1, 1, 1)]
void compute_main() {
  prevent_dce.Store4(0u, subgroupBallot_1a8251());
}

FXC validation failure:
c:\src\dawn\Shader@0x0000019446DA06A0(4,15-36): error X3004: undeclared identifier 'WaveActiveBallot'

