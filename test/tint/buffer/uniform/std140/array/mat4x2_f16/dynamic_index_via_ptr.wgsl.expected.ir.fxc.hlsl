SKIP: FAILED


cbuffer cbuffer_a : register(b0) {
  uint4 a[4];
};
RWByteAddressBuffer s : register(u1);
static int counter = 0;
int i() {
  counter = (counter + 1);
  return counter;
}

vector<float16_t, 2> tint_bitcast_to_f16(uint src) {
  uint v = src;
  float t_low = f16tof32((v & 65535u));
  float t_high = f16tof32(((v >> 16u) & 65535u));
  float16_t v_1 = float16_t(t_low);
  return vector<float16_t, 2>(v_1, float16_t(t_high));
}

matrix<float16_t, 4, 2> v_2(uint start_byte_offset) {
  uint4 v_3 = a[(start_byte_offset / 16u)];
  vector<float16_t, 2> v_4 = tint_bitcast_to_f16((((((start_byte_offset % 16u) / 4u) == 2u)) ? (v_3.z) : (v_3.x)));
  uint4 v_5 = a[((4u + start_byte_offset) / 16u)];
  vector<float16_t, 2> v_6 = tint_bitcast_to_f16(((((((4u + start_byte_offset) % 16u) / 4u) == 2u)) ? (v_5.z) : (v_5.x)));
  uint4 v_7 = a[((8u + start_byte_offset) / 16u)];
  vector<float16_t, 2> v_8 = tint_bitcast_to_f16(((((((8u + start_byte_offset) % 16u) / 4u) == 2u)) ? (v_7.z) : (v_7.x)));
  uint4 v_9 = a[((12u + start_byte_offset) / 16u)];
  return matrix<float16_t, 4, 2>(v_4, v_6, v_8, tint_bitcast_to_f16(((((((12u + start_byte_offset) % 16u) / 4u) == 2u)) ? (v_9.z) : (v_9.x))));
}

typedef matrix<float16_t, 4, 2> ary_ret[4];
ary_ret v_10(uint start_byte_offset) {
  matrix<float16_t, 4, 2> a[4] = (matrix<float16_t, 4, 2>[4])0;
  {
    uint v_11 = 0u;
    v_11 = 0u;
    while(true) {
      uint v_12 = v_11;
      if ((v_12 >= 4u)) {
        break;
      }
      a[v_12] = v_2((start_byte_offset + (v_12 * 16u)));
      {
        v_11 = (v_12 + 1u);
      }
      continue;
    }
  }
  matrix<float16_t, 4, 2> v_13[4] = a;
  return v_13;
}

[numthreads(1, 1, 1)]
void f() {
  uint v_14 = (16u * uint(i()));
  uint v_15 = (4u * uint(i()));
  matrix<float16_t, 4, 2> v_16[4] = v_10(0u);
  matrix<float16_t, 4, 2> l_a_i = v_2(v_14);
  uint4 v_17 = a[((v_14 + v_15) / 16u)];
  vector<float16_t, 2> l_a_i_i = tint_bitcast_to_f16(((((((v_14 + v_15) % 16u) / 4u) == 2u)) ? (v_17.z) : (v_17.x)));
  uint v_18 = a[((v_14 + v_15) / 16u)][(((v_14 + v_15) % 16u) / 4u)];
  matrix<float16_t, 4, 2> l_a[4] = v_16;
  s.Store<float16_t>(0u, (((float16_t(f16tof32((v_18 >> (((((v_14 + v_15) % 4u) == 0u)) ? (0u) : (16u))))) + l_a[0][0][0u]) + l_a_i[0][0u]) + l_a_i_i[0u]));
}

FXC validation failure:
c:\src\dawn\Shader@0x000002317E5024B0(12,8-16): error X3000: syntax error: unexpected token 'float16_t'

