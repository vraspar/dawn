SKIP: FAILED


static uint4 u = (1u).xxxx;
void f() {
  vector<float16_t, 4> v = vector<float16_t, 4>(u);
}

[numthreads(1, 1, 1)]
void unused_entry_point() {
}

FXC validation failure:
c:\src\dawn\Shader@0x000001E48E335290(4,10-18): error X3000: syntax error: unexpected token 'float16_t'

