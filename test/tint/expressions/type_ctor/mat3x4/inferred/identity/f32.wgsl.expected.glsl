#version 310 es

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void unused_entry_point() {
  return;
}
mat3x4 m = mat3x4(vec4(0.0f, 1.0f, 2.0f, 3.0f), vec4(4.0f, 5.0f, 6.0f, 7.0f), vec4(8.0f, 9.0f, 10.0f, 11.0f));
mat3x4 f() {
  mat3x4 m_1 = mat3x4(m);
  return m_1;
}

