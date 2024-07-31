
uint3 tint_mod_v3u32(uint3 lhs, uint3 rhs) {
  uint3 v = (((rhs == (0u).xxx)) ? ((1u).xxx) : (rhs));
  return (lhs - ((lhs / v) * v));
}

[numthreads(1, 1, 1)]
void f() {
  uint3 a = uint3(1u, 2u, 3u);
  uint3 b = uint3(4u, 5u, 6u);
  uint3 r = tint_mod_v3u32(a, b);
}

