#include "gpu/images.h"

namespace b2c {
namespace {
__device__ __forceinline__ uint32_t ch(uint32_t p, int c) { return (p >> (8 * c)) & 0xffu; }

__global__ void downsample_kernel(const uint32_t* __restrict__ rgba, const uint32_t* __restrict__ normals, const uint8_t* __restrict__ weights,
                                  int sw, int sh, int dw, int dh, uint32_t* __restrict__ o_rgba, uint32_t* __restrict__ o_normals, uint8_t* __restrict__ o_weights,
                                  float* __restrict__ normal_count) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dw || y >= dh) return;
  int x0 = min(2 * x, sw - 1), x1 = min(2 * x + 1, sw - 1), y0 = min(2 * y, sh - 1), y1 = min(2 * y + 1, sh - 1);
  int idx[4] = {y0 * sw + x0, y0 * sw + x1, y1 * sw + x0, y1 * sw + x1};
  uint32_t acc[4] = {0, 0, 0, 0};
  for (int k = 0; k < 4; k++) { uint32_t p = rgba[idx[k]]; for (int c = 0; c < 4; c++) acc[c] += ch(p, c); }
  uint32_t out = 0;
  for (int c = 0; c < 4; c++) out |= ((acc[c] + 2) / 4) << (8 * c);
  int o = y * dw + x;
  o_rgba[o] = out;
  float w = 1.f;
  if (weights) { uint32_t s = 0; for (int k = 0; k < 4; k++) s += weights[idx[k]]; o_weights[o] = (uint8_t)((s + 2) / 4); w = o_weights[o] * (1.f / 255.f); }
  if (normals) {
    uint32_t n = normals[idx[0]];
    o_normals[o] = n;
    if (((n >> 24) & 0xffu) > 127u) atomicAdd(normal_count, w);
  }
}
}  // namespace

void downsample_view(const ViewGPU& src, const DownsampleOut& dst, int dw, int dh, float* normal_count_accum, cudaStream_t stream) {
  dim3 block(16, 16), grid((dw + 15) / 16, (dh + 15) / 16);
  downsample_kernel<<<grid, block, 0, stream>>>(src.rgba, src.normals, src.weights, src.W, src.H, dw, dh, dst.rgba, dst.normals, dst.weights, normal_count_accum);
  CUDA_KERNEL_CHECK();
}
}  // namespace b2c
