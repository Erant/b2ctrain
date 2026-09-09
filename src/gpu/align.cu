#include "gpu/align.h"
#include <cmath>
#include <algorithm>

namespace b2c {
namespace {

constexpr int HIST_BINS = 4096;
constexpr float HIST_MAX = 64.f;      // px; magnitudes beyond land in the last bin
constexpr int FG_ALPHA = 32;          // alpha (of 255) above which a pixel is subject (align.py's _FOREGROUND_ALPHA)
constexpr int LK_RADIUS = 4;          // window half-size at every level
constexpr int LK_ITERS = 4;           // Gauss-Newton iterations per level
constexpr float LK_DAMP = 4000.f;     // absolute Levenberg damping (see lk_kernel)
constexpr float LK_SMOOTH_SIGMA = 2.0f;  // flow smoothing between iterations, px (propagates texture to flat areas)

__device__ __forceinline__ float gray(float r, float g, float b) { return 0.299f * r + 0.587f * g + 0.114f * b; }

// Training frame composited over grey 0.5 (both sides of the flow are flattened onto the same grey, align.py).
__global__ void gray_frame_kernel(const uint32_t* __restrict__ rgba, int n, float* __restrict__ out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t p = rgba[i];
  float a = (float)(p >> 24) * (1.f / 255.f);
  float r = (float)(p & 0xffu) + (1.f - a) * 127.5f, g = (float)((p >> 8) & 0xffu) + (1.f - a) * 127.5f, b = (float)((p >> 16) & 0xffu) + (1.f - a) * 127.5f;
  out[i] = gray(r, g, b);
}
__global__ void gray_render_kernel(const float4* __restrict__ rgba, int n, float* __restrict__ out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 c = rgba[i];
  out[i] = gray(fminf(fmaxf(c.x, 0.f), 1.f) * 255.f, fminf(fmaxf(c.y, 0.f), 1.f) * 255.f, fminf(fmaxf(c.z, 0.f), 1.f) * 255.f);
}
__global__ void down2_kernel(const float* __restrict__ src, int sw, int sh, float* __restrict__ dst, int dw, int dh) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dw || y >= dh) return;
  int x0 = min(2 * x, sw - 1), x1 = min(2 * x + 1, sw - 1), y0 = min(2 * y, sh - 1), y1 = min(2 * y + 1, sh - 1);
  dst[y * dw + x] = 0.25f * (src[y0 * sw + x0] + src[y0 * sw + x1] + src[y1 * sw + x0] + src[y1 * sw + x1]);
}
__global__ void grad_kernel(const float* __restrict__ img, int w, int h, float* __restrict__ gx, float* __restrict__ gy) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  int xm = max(x - 1, 0), xp = min(x + 1, w - 1), ym = max(y - 1, 0), yp = min(y + 1, h - 1);
  gx[y * w + x] = 0.5f * (img[y * w + xp] - img[y * w + xm]);
  gy[y * w + x] = 0.5f * (img[yp * w + x] - img[ym * w + x]);
}
__device__ __forceinline__ float bilerp(const float* __restrict__ img, int w, int h, float x, float y) {
  x = fminf(fmaxf(x, 0.f), (float)(w - 1)); y = fminf(fmaxf(y, 0.f), (float)(h - 1));
  int x0 = (int)x, y0 = (int)y; int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
  float fx = x - x0, fy = y - y0;
  float a = img[y0 * w + x0], b = img[y0 * w + x1], c = img[y1 * w + x0], d = img[y1 * w + x1];
  return (a * (1.f - fx) + b * fx) * (1.f - fy) + (c * (1.f - fx) + d * fx) * fy;
}
// One Gauss-Newton step of windowed Lucas-Kanade: A (frame) at integer positions vs B (render) sampled at q + flow.
__global__ void lk_kernel(const float* __restrict__ A, const float* __restrict__ B, const float* __restrict__ Bx, const float* __restrict__ By,
                          int w, int h, const float2* __restrict__ flow_in, float2* __restrict__ flow_out) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  float2 u = flow_in[y * w + x];
  float gxx = 0.f, gxy = 0.f, gyy = 0.f, bx = 0.f, by = 0.f;
  for (int dy = -LK_RADIUS; dy <= LK_RADIUS; dy++) {
    int qy = y + dy; if (qy < 0 || qy >= h) continue;
    for (int dx = -LK_RADIUS; dx <= LK_RADIUS; dx++) {
      int qx = x + dx; if (qx < 0 || qx >= w) continue;
      float sx = qx + u.x, sy = qy + u.y;
      if (sx < 0.f || sy < 0.f || sx > w - 1 || sy > h - 1) continue;
      float a = A[qy * w + qx];
      float b = bilerp(B, w, h, sx, sy), ix = bilerp(Bx, w, h, sx, sy), iy = bilerp(By, w, h, sx, sy);
      float it = b - a;
      gxx += ix * ix; gxy += ix * iy; gyy += iy * iy; bx += ix * it; by += iy * it;
    }
  }
  // Levenberg damping, absolute as well as relative: a window with little texture (skin, flat cloth) has a small,
  // noisy normal matrix and would otherwise report a large displacement fitted to nothing; LK_DAMP is a gradient
  // energy well below any real edge's (sum of squared gradients over the window, gray in 0..255).
  float lambda = 1e-3f * (gxx + gyy) + LK_DAMP;
  float det = (gxx + lambda) * (gyy + lambda) - gxy * gxy;
  float2 du = make_float2(0.f, 0.f);
  if (det > 1e-12f) {
    du.x = -((gyy + lambda) * bx - gxy * by) / det;
    du.y = -(-gxy * bx + (gxx + lambda) * by) / det;
    float m = sqrtf(du.x * du.x + du.y * du.y);
    if (m > 1.f) { du.x /= m; du.y /= m; }  // at most one pixel per iteration
    if (!isfinite(du.x) || !isfinite(du.y)) du = make_float2(0.f, 0.f);
  }
  flow_out[y * w + x] = make_float2(u.x + du.x, u.y + du.y);
}
__global__ void upsample_flow_kernel(const float2* __restrict__ src, int sw, int sh, float2* __restrict__ dst, int dw, int dh) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dw || y >= dh) return;
  float sx = (x + 0.5f) * 0.5f - 0.5f, sy = (y + 0.5f) * 0.5f - 0.5f;
  sx = fminf(fmaxf(sx, 0.f), (float)(sw - 1)); sy = fminf(fmaxf(sy, 0.f), (float)(sh - 1));
  int x0 = (int)sx, y0 = (int)sy, x1 = min(x0 + 1, sw - 1), y1 = min(y0 + 1, sh - 1);
  float fx = sx - x0, fy = sy - y0;
  float2 a = src[y0 * sw + x0], b = src[y0 * sw + x1], c = src[y1 * sw + x0], d = src[y1 * sw + x1];
  float2 v = make_float2(((a.x * (1.f - fx) + b.x * fx) * (1.f - fy) + (c.x * (1.f - fx) + d.x * fx) * fy) * 2.f,
                         ((a.y * (1.f - fx) + b.y * fx) * (1.f - fy) + (c.y * (1.f - fx) + d.y * fx) * fy) * 2.f);
  dst[y * dw + x] = v;
}
__global__ void zero_outside_kernel(const uint32_t* __restrict__ rgba, int n, float2* __restrict__ flow) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if ((rgba[i] >> 24) <= (uint32_t)FG_ALPHA) flow[i] = make_float2(0.f, 0.f);
}
// Separable Gaussian with OpenCV's kernel size int(3 sigma) | 1 and BORDER_REFLECT_101.
__device__ __forceinline__ int reflect101(int i, int n) { if (n == 1) return 0; while (i < 0 || i >= n) { if (i < 0) i = -i; if (i >= n) i = 2 * n - 2 - i; } return i; }
__global__ void blur_kernel(const float2* __restrict__ src, int w, int h, const float* __restrict__ k, int radius, bool horizontal, float2* __restrict__ dst) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  float2 acc = make_float2(0.f, 0.f);
  for (int t = -radius; t <= radius; t++) {
    int sx = horizontal ? reflect101(x + t, w) : x, sy = horizontal ? y : reflect101(y + t, h);
    float2 v = src[sy * w + sx]; float wt = k[t + radius];
    acc.x += v.x * wt; acc.y += v.y * wt;
  }
  dst[y * w + x] = acc;
}
__global__ void cap_stats_kernel(const uint32_t* __restrict__ rgba, int n, float cap, float2* __restrict__ flow, float* __restrict__ hist) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float2 f = flow[i];
  float m = sqrtf(f.x * f.x + f.y * f.y);
  if ((rgba[i] >> 24) > (uint32_t)FG_ALPHA) {
    int bin = min((int)(m * (HIST_BINS / HIST_MAX)), HIST_BINS - 1);
    atomicAdd(hist + bin, 1.f); atomicAdd(hist + HIST_BINS, m); atomicAdd(hist + HIST_BINS + 1, 1.f);
  }
  float s = fminf(1.f, cap / (m + 1e-6f));
  flow[i] = make_float2(f.x * s, f.y * s);
}
__device__ __forceinline__ float lanczos4(float z) {
  z = fabsf(z);
  if (z < 1e-5f) return 1.f;
  if (z >= 4.f) return 0.f;
  float pz = 3.14159265f * z;
  return 4.f * sinf(pz) * sinf(pz * 0.25f) / (pz * pz);
}
// Backward warp dst(x, y) = src(x - u, y - v), Lanczos-4, constant (transparent) border, on packed RGBA8.
__global__ void warp_kernel(const uint32_t* __restrict__ src, int w, int h, const float2* __restrict__ flow, uint32_t* __restrict__ dst) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  float2 f = flow[y * w + x];
  float sx = x - f.x, sy = y - f.y;
  if (f.x == 0.f && f.y == 0.f) { dst[y * w + x] = src[y * w + x]; return; }  // exact fixed point
  int ix = (int)floorf(sx), iy = (int)floorf(sy);
  float fx = sx - ix, fy = sy - iy;
  float wx[8], wy[8], sxw = 0.f, syw = 0.f;
#pragma unroll
  for (int t = 0; t < 8; t++) { wx[t] = lanczos4(fx - (t - 3)); wy[t] = lanczos4(fy - (t - 3)); sxw += wx[t]; syw += wy[t]; }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int ty = 0; ty < 8; ty++) {
    int yy = iy + ty - 3; if (yy < 0 || yy >= h) continue;
    for (int tx = 0; tx < 8; tx++) {
      int xx = ix + tx - 3; if (xx < 0 || xx >= w) continue;
      uint32_t p = src[yy * w + xx]; float wt = wx[tx] * wy[ty];
#pragma unroll
      for (int c = 0; c < 4; c++) acc[c] += wt * (float)((p >> (8 * c)) & 0xffu);
    }
  }
  float norm = 1.f / (sxw * syw);
  uint32_t out = 0;
#pragma unroll
  for (int c = 0; c < 4; c++) { float v = fminf(fmaxf(acc[c] * norm + 0.5f, 0.f), 255.f); out |= ((uint32_t)v) << (8 * c); }
  dst[y * w + x] = out;
}

// Box average of the field over ALIGN_WARP_DOWN x ALIGN_WARP_DOWN blocks (edge blocks over what exists).
__global__ void down_flow_kernel(const float2* __restrict__ src, int w, int h, float2* __restrict__ dst, int dw, int dh) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dw || y >= dh) return;
  float2 acc = make_float2(0.f, 0.f); int c = 0;
  for (int dy = 0; dy < ALIGN_WARP_DOWN; dy++) { int sy = y * ALIGN_WARP_DOWN + dy; if (sy >= h) break;
    for (int dx = 0; dx < ALIGN_WARP_DOWN; dx++) { int sx = x * ALIGN_WARP_DOWN + dx; if (sx >= w) break; float2 v = src[sy * w + sx]; acc.x += v.x; acc.y += v.y; c++; } }
  dst[y * dw + x] = c ? make_float2(acc.x / c, acc.y / c) : make_float2(0.f, 0.f);
}

dim3 grid2(int w, int h) { return dim3((w + 15) / 16, (h + 15) / 16); }

}  // namespace

void AlignScratch::setup(int W, int H) {
  levels = 0;
  int w = W, h = H;
  for (int l = 0; l < ALIGN_LEVELS; l++) {
    if (l > 0 && (w < 32 || h < 32)) break;
    lw[l] = w; lh[l] = h; levels = l + 1;
    size_t n = (size_t)w * h;
    ga[l].reserve(n); gb[l].reserve(n); gbx[l].reserve(n); gby[l].reserve(n);
    w = (w + 1) / 2; h = (h + 1) / 2;
  }
  flow.reserve((size_t)W * H); flow_prev.reserve((size_t)W * H); tmp.reserve((size_t)W * H);
  hist.reserve(HIST_BINS + 2); h_hist.reserve(HIST_BINS + 2);
}

AlignStats align_view_gpu(AlignScratch& s, const uint32_t* frame, const float4* render, int W, int H, float sigma, float cap,
                          uint32_t* dst, cudaStream_t stream, float2* warp_out) {
  if (s.levels == 0 || s.lw[0] != W || s.lh[0] != H) s.setup(W, H);
  const int n = W * H;
  gray_frame_kernel<<<div_up(n, 256), 256, 0, stream>>>(frame, n, s.ga[0]);
  gray_render_kernel<<<div_up(n, 256), 256, 0, stream>>>(render, n, s.gb[0]);
  for (int l = 1; l < s.levels; l++) {
    down2_kernel<<<grid2(s.lw[l], s.lh[l]), dim3(16, 16), 0, stream>>>(s.ga[l - 1], s.lw[l - 1], s.lh[l - 1], s.ga[l], s.lw[l], s.lh[l]);
    down2_kernel<<<grid2(s.lw[l], s.lh[l]), dim3(16, 16), 0, stream>>>(s.gb[l - 1], s.lw[l - 1], s.lh[l - 1], s.gb[l], s.lw[l], s.lh[l]);
  }
  for (int l = 0; l < s.levels; l++) grad_kernel<<<grid2(s.lw[l], s.lh[l]), dim3(16, 16), 0, stream>>>(s.gb[l], s.lw[l], s.lh[l], s.gbx[l], s.gby[l]);
  CUDA_KERNEL_CHECK();
  // Coarse-to-fine Lucas-Kanade; `flow` holds the current estimate at the current level. A small blur between
  // iterations propagates estimates from textured pixels into their flat neighbours (the role DIS's variational
  // refinement plays), so the field the final sigma blur sees is coherent rather than per-pixel noise.
  if (s.smooth_k.count == 0) {
    int ksize = ((int)(LK_SMOOTH_SIGMA * 3.f)) | 1, radius = ksize / 2;
    std::vector<float> k(ksize); double sum = 0;
    for (int t = -radius; t <= radius; t++) { double v = std::exp(-(double)t * t / (2.0 * LK_SMOOTH_SIGMA * LK_SMOOTH_SIGMA)); k[t + radius] = (float)v; sum += v; }
    for (auto& v : k) v = (float)(v / sum);
    s.smooth_k.upload(k, stream); s.smooth_radius = radius;
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  for (int l = s.levels - 1; l >= 0; l--) {
    int w = s.lw[l], h = s.lh[l];
    if (l == s.levels - 1) CUDA_CHECK(cudaMemsetAsync(s.flow.ptr, 0, (size_t)w * h * sizeof(float2), stream));
    else upsample_flow_kernel<<<grid2(w, h), dim3(16, 16), 0, stream>>>(s.flow_prev, s.lw[l + 1], s.lh[l + 1], s.flow, w, h);
    for (int it = 0; it < LK_ITERS; it++) {
      lk_kernel<<<grid2(w, h), dim3(16, 16), 0, stream>>>(s.ga[l], s.gb[l], s.gbx[l], s.gby[l], w, h, s.flow, s.tmp);
      blur_kernel<<<grid2(w, h), dim3(16, 16), 0, stream>>>(s.tmp, w, h, s.smooth_k, s.smooth_radius, true, s.flow);
      blur_kernel<<<grid2(w, h), dim3(16, 16), 0, stream>>>(s.flow, w, h, s.smooth_k, s.smooth_radius, false, s.tmp);
      std::swap(s.flow, s.tmp);
    }
    if (l > 0) std::swap(s.flow, s.flow_prev);
  }
  CUDA_KERNEL_CHECK();
  zero_outside_kernel<<<div_up(n, 256), 256, 0, stream>>>(frame, n, s.flow);
  // Gaussian blur with OpenCV's truncated kernel: ksize = int(3 sigma) | 1.
  {
    int ksize = std::max(1, ((int)(sigma * 3.f)) | 1), radius = ksize / 2;
    std::vector<float> k(ksize); double sum = 0;
    for (int t = -radius; t <= radius; t++) { double v = std::exp(-(double)t * t / (2.0 * sigma * sigma)); k[t + radius] = (float)v; sum += v; }
    for (auto& v : k) v = (float)(v / sum);
    DevBuf<float> dk; dk.upload(k, stream);
    blur_kernel<<<grid2(W, H), dim3(16, 16), 0, stream>>>(s.flow, W, H, dk, radius, true, s.tmp);
    blur_kernel<<<grid2(W, H), dim3(16, 16), 0, stream>>>(s.tmp, W, H, dk, radius, false, s.flow);
    CUDA_CHECK(cudaStreamSynchronize(stream));  // dk is freed on return
  }
  s.hist.zero(stream);
  cap_stats_kernel<<<div_up(n, 256), 256, 0, stream>>>(frame, n, cap, s.flow, s.hist);
  if (dst) warp_kernel<<<grid2(W, H), dim3(16, 16), 0, stream>>>(frame, W, H, s.flow, dst);
  if (warp_out) { int dw = align_warp_dim(W), dh = align_warp_dim(H); down_flow_kernel<<<grid2(dw, dh), dim3(16, 16), 0, stream>>>(s.flow, W, H, warp_out, dw, dh); }
  CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaMemcpyAsync(s.h_hist.ptr, s.hist.ptr, (HIST_BINS + 2) * sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  AlignStats st;
  float count = s.h_hist.ptr[HIST_BINS + 1];
  if (count > 0.f) {
    st.mean = s.h_hist.ptr[HIST_BINS] / count;
    float target = 0.9f * count, acc = 0.f; int bin = 0;
    for (; bin < HIST_BINS; bin++) { acc += s.h_hist.ptr[bin]; if (acc >= target) break; }
    st.p90 = (bin + 0.5f) * (HIST_MAX / HIST_BINS);
  }
  return st;
}

}  // namespace b2c
