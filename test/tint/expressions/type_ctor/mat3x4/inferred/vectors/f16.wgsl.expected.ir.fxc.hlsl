SKIP: FAILED


static matrix<float16_t, 3, 4> m = matrix<float16_t, 3, 4>(vector<float16_t, 4>(float16_t(0.0h), float16_t(1.0h), float16_t(2.0h), float16_t(3.0h)), vector<float16_t, 4>(float16_t(4.0h), float16_t(5.0h), float16_t(6.0h), float16_t(7.0h)), vector<float16_t, 4>(float16_t(8.0h), float16_t(9.0h), float16_t(10.0h), float16_t(11.0h)));
RWByteAddressBuffer tint_symbol : register(u0);
void v(uint offset, matrix<float16_t, 3, 4> obj) {
  tint_symbol.Store<vector<float16_t, 4> >((offset + 0u), obj[0u]);
  tint_symbol.Store<vector<float16_t, 4> >((offset + 8u), obj[1u]);
  tint_symbol.Store<vector<float16_t, 4> >((offset + 16u), obj[2u]);
}

[numthreads(1, 1, 1)]
void f() {
  v(0u, m);
}

FXC validation failure:
c:\src\dawn\Shader@0x000001FEF3DFE2F0(2,15-23): error X3000: syntax error: unexpected token 'float16_t'

