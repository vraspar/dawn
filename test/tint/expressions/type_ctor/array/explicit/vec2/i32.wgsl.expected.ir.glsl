#version 310 es

int arr[2] = int[2](ivec2(1), ivec2(2));
void f() {
  int v[2] = arr;
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
}
