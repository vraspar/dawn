#version 310 es
#extension GL_AMD_gpu_shader_half_float: require


struct S {
  f16vec3 v;
};

S P = S(f16vec3(0.0hf));
void f() {
  f16vec3 v = P.v;
  float16_t x = P.v.x;
  float16_t y = P.v.y;
  float16_t z = P.v.z;
  f16vec2 xx = P.v.xx;
  f16vec2 xy = P.v.xy;
  f16vec2 xz = P.v.xz;
  f16vec2 yx = P.v.yx;
  f16vec2 yy = P.v.yy;
  f16vec2 yz = P.v.yz;
  f16vec2 zx = P.v.zx;
  f16vec2 zy = P.v.zy;
  f16vec2 zz = P.v.zz;
  f16vec3 xxx = P.v.xxx;
  f16vec3 xxy = P.v.xxy;
  f16vec3 xxz = P.v.xxz;
  f16vec3 xyx = P.v.xyx;
  f16vec3 xyy = P.v.xyy;
  f16vec3 xyz = P.v.xyz;
  f16vec3 xzx = P.v.xzx;
  f16vec3 xzy = P.v.xzy;
  f16vec3 xzz = P.v.xzz;
  f16vec3 yxx = P.v.yxx;
  f16vec3 yxy = P.v.yxy;
  f16vec3 yxz = P.v.yxz;
  f16vec3 yyx = P.v.yyx;
  f16vec3 yyy = P.v.yyy;
  f16vec3 yyz = P.v.yyz;
  f16vec3 yzx = P.v.yzx;
  f16vec3 yzy = P.v.yzy;
  f16vec3 yzz = P.v.yzz;
  f16vec3 zxx = P.v.zxx;
  f16vec3 zxy = P.v.zxy;
  f16vec3 zxz = P.v.zxz;
  f16vec3 zyx = P.v.zyx;
  f16vec3 zyy = P.v.zyy;
  f16vec3 zyz = P.v.zyz;
  f16vec3 zzx = P.v.zzx;
  f16vec3 zzy = P.v.zzy;
  f16vec3 zzz = P.v.zzz;
  f16vec4 xxxx = P.v.xxxx;
  f16vec4 xxxy = P.v.xxxy;
  f16vec4 xxxz = P.v.xxxz;
  f16vec4 xxyx = P.v.xxyx;
  f16vec4 xxyy = P.v.xxyy;
  f16vec4 xxyz = P.v.xxyz;
  f16vec4 xxzx = P.v.xxzx;
  f16vec4 xxzy = P.v.xxzy;
  f16vec4 xxzz = P.v.xxzz;
  f16vec4 xyxx = P.v.xyxx;
  f16vec4 xyxy = P.v.xyxy;
  f16vec4 xyxz = P.v.xyxz;
  f16vec4 xyyx = P.v.xyyx;
  f16vec4 xyyy = P.v.xyyy;
  f16vec4 xyyz = P.v.xyyz;
  f16vec4 xyzx = P.v.xyzx;
  f16vec4 xyzy = P.v.xyzy;
  f16vec4 xyzz = P.v.xyzz;
  f16vec4 xzxx = P.v.xzxx;
  f16vec4 xzxy = P.v.xzxy;
  f16vec4 xzxz = P.v.xzxz;
  f16vec4 xzyx = P.v.xzyx;
  f16vec4 xzyy = P.v.xzyy;
  f16vec4 xzyz = P.v.xzyz;
  f16vec4 xzzx = P.v.xzzx;
  f16vec4 xzzy = P.v.xzzy;
  f16vec4 xzzz = P.v.xzzz;
  f16vec4 yxxx = P.v.yxxx;
  f16vec4 yxxy = P.v.yxxy;
  f16vec4 yxxz = P.v.yxxz;
  f16vec4 yxyx = P.v.yxyx;
  f16vec4 yxyy = P.v.yxyy;
  f16vec4 yxyz = P.v.yxyz;
  f16vec4 yxzx = P.v.yxzx;
  f16vec4 yxzy = P.v.yxzy;
  f16vec4 yxzz = P.v.yxzz;
  f16vec4 yyxx = P.v.yyxx;
  f16vec4 yyxy = P.v.yyxy;
  f16vec4 yyxz = P.v.yyxz;
  f16vec4 yyyx = P.v.yyyx;
  f16vec4 yyyy = P.v.yyyy;
  f16vec4 yyyz = P.v.yyyz;
  f16vec4 yyzx = P.v.yyzx;
  f16vec4 yyzy = P.v.yyzy;
  f16vec4 yyzz = P.v.yyzz;
  f16vec4 yzxx = P.v.yzxx;
  f16vec4 yzxy = P.v.yzxy;
  f16vec4 yzxz = P.v.yzxz;
  f16vec4 yzyx = P.v.yzyx;
  f16vec4 yzyy = P.v.yzyy;
  f16vec4 yzyz = P.v.yzyz;
  f16vec4 yzzx = P.v.yzzx;
  f16vec4 yzzy = P.v.yzzy;
  f16vec4 yzzz = P.v.yzzz;
  f16vec4 zxxx = P.v.zxxx;
  f16vec4 zxxy = P.v.zxxy;
  f16vec4 zxxz = P.v.zxxz;
  f16vec4 zxyx = P.v.zxyx;
  f16vec4 zxyy = P.v.zxyy;
  f16vec4 zxyz = P.v.zxyz;
  f16vec4 zxzx = P.v.zxzx;
  f16vec4 zxzy = P.v.zxzy;
  f16vec4 zxzz = P.v.zxzz;
  f16vec4 zyxx = P.v.zyxx;
  f16vec4 zyxy = P.v.zyxy;
  f16vec4 zyxz = P.v.zyxz;
  f16vec4 zyyx = P.v.zyyx;
  f16vec4 zyyy = P.v.zyyy;
  f16vec4 zyyz = P.v.zyyz;
  f16vec4 zyzx = P.v.zyzx;
  f16vec4 zyzy = P.v.zyzy;
  f16vec4 zyzz = P.v.zyzz;
  f16vec4 zzxx = P.v.zzxx;
  f16vec4 zzxy = P.v.zzxy;
  f16vec4 zzxz = P.v.zzxz;
  f16vec4 zzyx = P.v.zzyx;
  f16vec4 zzyy = P.v.zzyy;
  f16vec4 zzyz = P.v.zzyz;
  f16vec4 zzzx = P.v.zzzx;
  f16vec4 zzzy = P.v.zzzy;
  f16vec4 zzzz = P.v.zzzz;
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
}
