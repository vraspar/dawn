
RWByteAddressBuffer prevent_dce : register(u0);
RWByteAddressBuffer sb_rw : register(u1);
uint atomicAdd_8a199a() {
  uint v = 0u;
  sb_rw.InterlockedAdd(uint(0u), 1u, v);
  uint res = v;
  return res;
}

void fragment_main() {
  prevent_dce.Store(0u, atomicAdd_8a199a());
}

[numthreads(1, 1, 1)]
void compute_main() {
  prevent_dce.Store(0u, atomicAdd_8a199a());
}

