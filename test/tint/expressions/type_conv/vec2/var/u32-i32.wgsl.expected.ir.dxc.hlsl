
static uint2 u = (1u).xx;
void f() {
  int2 v = int2(u);
}

[numthreads(1, 1, 1)]
void unused_entry_point() {
}

