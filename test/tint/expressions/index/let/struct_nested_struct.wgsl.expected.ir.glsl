#version 310 es


struct T {
  float o;
  uint p;
};

struct S {
  int m;
  T n;
};

uint f() {
  S a = S(0, T(0.0f, 0u));
  return a.n.p;
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
}
