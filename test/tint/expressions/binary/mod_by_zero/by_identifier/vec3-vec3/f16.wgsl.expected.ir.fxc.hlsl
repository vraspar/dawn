SKIP: FAILED


[numthreads(1, 1, 1)]
void f() {
  vector<float16_t, 3> a = vector<float16_t, 3>(float16_t(1.0h), float16_t(2.0h), float16_t(3.0h));
  vector<float16_t, 3> b = vector<float16_t, 3>(float16_t(0.0h), float16_t(5.0h), float16_t(0.0h));
  vector<float16_t, 3> v = a;
  vector<float16_t, 3> v_1 = b;
  vector<float16_t, 3> v_2 = (v / v_1);
  vector<float16_t, 3> v_3 = floor(v_2);
  vector<float16_t, 3> r = ((v - (((v_2 < (float16_t(0.0h)).xxx)) ? (ceil(v_2)) : (v_3))) * v_1);
}

FXC validation failure:
c:\src\dawn\Shader@0x000001F4B9E25C40(4,10-18): error X3000: syntax error: unexpected token 'float16_t'
c:\src\dawn\Shader@0x000001F4B9E25C40(5,10-18): error X3000: syntax error: unexpected token 'float16_t'
c:\src\dawn\Shader@0x000001F4B9E25C40(6,10-18): error X3000: syntax error: unexpected token 'float16_t'
c:\src\dawn\Shader@0x000001F4B9E25C40(7,10-18): error X3000: syntax error: unexpected token 'float16_t'
c:\src\dawn\Shader@0x000001F4B9E25C40(8,10-18): error X3000: syntax error: unexpected token 'float16_t'
c:\src\dawn\Shader@0x000001F4B9E25C40(9,10-18): error X3000: syntax error: unexpected token 'float16_t'
c:\src\dawn\Shader@0x000001F4B9E25C40(10,10-18): error X3000: syntax error: unexpected token 'float16_t'

