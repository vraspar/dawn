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

