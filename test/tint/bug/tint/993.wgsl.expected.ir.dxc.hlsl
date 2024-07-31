
cbuffer cbuffer_constants : register(b0, space1) {
  uint4 constants[1];
};
RWByteAddressBuffer result : register(u1, space1);
RWByteAddressBuffer s : register(u0);
int runTest() {
  int v = 0;
  s.InterlockedOr(int(0u), 0, v);
  return v;
}

[numthreads(1, 1, 1)]
void main() {
  result.Store(0u, uint(runTest()));
}

