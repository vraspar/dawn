SKIP: FAILED

struct InnerS {
  int v;
};

struct OuterS {
  InnerS a1[8];
};


cbuffer cbuffer_uniforms : register(b4, space1) {
  uint4 uniforms[1];
};
[numthreads(1, 1, 1)]
void main() {
  InnerS v = (InnerS)0;
  OuterS s1 = (OuterS)0;
  {
    int i = 0;
    while(true) {
      if ((i < 4)) {
      } else {
        break;
      }
      InnerS v_1 = v;
      s1.a1[uniforms[0u].x] = v_1;
      {
        i = (i + 1);
      }
      continue;
    }
  }
}

FXC validation failure:
c:\src\dawn\Shader@0x000001DB602A4100(25,7-27): error X3500: array reference cannot be used as an l-value; not natively addressable
c:\src\dawn\Shader@0x000001DB602A4100(19,5-15): error X3511: forced to unroll loop, but unrolling failed.

