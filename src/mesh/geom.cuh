#pragma once
// Device geometry shared by the mesh kernels.
#include "gpu/util.cuh"

namespace b2c {

// Closest point on triangle abc to p (Ericson, Real-Time Collision Detection 5.1.5).
// region: 0 face, 1..3 vertex a/b/c, 4..6 edge ab/bc/ca. bary = (wa, wb, wc).
__device__ __forceinline__ float3 closest_on_tri(float3 p, float3 a, float3 b, float3 c, float3& bary, int& region) {
  float3 ab = b - a, ac = c - a, ap = p - a;
  float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
  if (d1 <= 0.f && d2 <= 0.f) { bary = make_float3(1, 0, 0); region = 1; return a; }
  float3 bp = p - b; float d3 = dot3(ab, bp), d4 = dot3(ac, bp);
  if (d3 >= 0.f && d4 <= d3) { bary = make_float3(0, 1, 0); region = 2; return b; }
  float vc = d1 * d4 - d3 * d2;
  if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) { float t = d1 / (d1 - d3); bary = make_float3(1 - t, t, 0); region = 4; return a + ab * t; }
  float3 cp = p - c; float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
  if (d6 >= 0.f && d5 <= d6) { bary = make_float3(0, 0, 1); region = 3; return c; }
  float vb = d5 * d2 - d1 * d6;
  if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) { float t = d2 / (d2 - d6); bary = make_float3(1 - t, 0, t); region = 6; return a + ac * t; }
  float va = d3 * d6 - d5 * d4;
  if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f) { float t = (d4 - d3) / ((d4 - d3) + (d5 - d6)); bary = make_float3(0, 1 - t, t); region = 5; return b + (c - b) * t; }
  float denom = 1.f / (va + vb + vc); float v = vb * denom, w = vc * denom;
  bary = make_float3(1 - v - w, v, w); region = 0; return a + ab * v + ac * w;
}

// Bilinear sample of a W x H x C uint8 image at continuous pixel coordinates where integer coordinates are pixel
// centres (cv2.remap's convention); zero outside (BORDER_CONSTANT).
__device__ __forceinline__ float sample_bilinear_u8(const uint8_t* img, int W, int H, int C, int ch, float x, float y) {
  int x0 = (int)floorf(x), y0 = (int)floorf(y); float fx = x - x0, fy = y - y0;
  auto at = [&](int xx, int yy) -> float { return (xx < 0 || yy < 0 || xx >= W || yy >= H) ? 0.f : (float)img[((size_t)yy * W + xx) * C + ch]; };
  return (1 - fx) * (1 - fy) * at(x0, y0) + fx * (1 - fy) * at(x0 + 1, y0) + (1 - fx) * fy * at(x0, y0 + 1) + fx * fy * at(x0 + 1, y0 + 1);
}
__device__ __forceinline__ float sample_bilinear_f(const float* img, int W, int H, float x, float y) {
  int x0 = (int)floorf(x), y0 = (int)floorf(y); float fx = x - x0, fy = y - y0;
  auto at = [&](int xx, int yy) -> float { return (xx < 0 || yy < 0 || xx >= W || yy >= H) ? 0.f : img[(size_t)yy * W + xx]; };
  return (1 - fx) * (1 - fy) * at(x0, y0) + fx * (1 - fy) * at(x0 + 1, y0) + (1 - fx) * fy * at(x0, y0 + 1) + fx * fy * at(x0 + 1, y0 + 1);
}

}  // namespace b2c
