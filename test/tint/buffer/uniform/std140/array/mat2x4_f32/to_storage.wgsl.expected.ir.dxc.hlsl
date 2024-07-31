
cbuffer cbuffer_u : register(b0) {
  uint4 u[8];
};
RWByteAddressBuffer s : register(u1);
void v(uint offset, float2x4 obj) {
  s.Store4((offset + 0u), asuint(obj[0u]));
  s.Store4((offset + 16u), asuint(obj[1u]));
}

float2x4 v_1(uint start_byte_offset) {
  float4 v_2 = asfloat(u[(start_byte_offset / 16u)]);
  return float2x4(v_2, asfloat(u[((16u + start_byte_offset) / 16u)]));
}

void v_3(uint offset, float2x4 obj[4]) {
  {
    uint v_4 = 0u;
    v_4 = 0u;
    while(true) {
      uint v_5 = v_4;
      if ((v_5 >= 4u)) {
        break;
      }
      v((offset + (v_5 * 32u)), obj[v_5]);
      {
        v_4 = (v_5 + 1u);
      }
      continue;
    }
  }
}

typedef float2x4 ary_ret[4];
ary_ret v_6(uint start_byte_offset) {
  float2x4 a[4] = (float2x4[4])0;
  {
    uint v_7 = 0u;
    v_7 = 0u;
    while(true) {
      uint v_8 = v_7;
      if ((v_8 >= 4u)) {
        break;
      }
      a[v_8] = v_1((start_byte_offset + (v_8 * 32u)));
      {
        v_7 = (v_8 + 1u);
      }
      continue;
    }
  }
  float2x4 v_9[4] = a;
  return v_9;
}

[numthreads(1, 1, 1)]
void f() {
  float2x4 v_10[4] = v_6(0u);
  v_3(0u, v_10);
  v(32u, v_1(64u));
  s.Store4(32u, asuint(asfloat(u[1u]).ywxz));
  s.Store(32u, asuint(asfloat(u[1u].x)));
}

