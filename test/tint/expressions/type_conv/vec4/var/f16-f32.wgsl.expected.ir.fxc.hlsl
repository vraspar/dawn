SKIP: FAILED


static vector<float16_t, 4> u = (float16_t(1.0h)).xxxx;
void f() {
  float4 v = float4(u);
}

[numthreads(1, 1, 1)]
void unused_entry_point() {
}

FXC validation failure:
c:\src\dawn\Shader@0x000001AE1E855290(2,15-23): error X3000: syntax error: unexpected token 'float16_t'

