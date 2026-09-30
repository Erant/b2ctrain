#include "gpu/pose_contain.h"

namespace b2c {

namespace {

__global__ void hide_kernel(int n, float4* __restrict__ pos, const uint8_t* __restrict__ deep) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && deep[i]) pos[i].w = -30.f;
}

__global__ void target_kernel(const float4* __restrict__ img, int W, int H, int r, float thr, uint32_t* __restrict__ rgba, uint8_t* __restrict__ w) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  bool in = false;
  for (int dy = -r; dy <= r && !in; dy++) {
    int yy = y + dy; if (yy < 0 || yy >= H) continue;
    for (int dx = -r; dx <= r; dx++) {
      int xx = x + dx; if (xx < 0 || xx >= W || dx * dx + dy * dy > r * r) continue;
      if (img[yy * W + xx].w > thr) { in = true; break; }
    }
  }
  rgba[y * W + x] = in ? 0xff000000u : 0u;
  w[y * W + x] = in ? 0 : 255;
}

}  // namespace

void pose_contain_hide(float4* pos_view, const uint8_t* deep, int n, cudaStream_t stream) {
  hide_kernel<<<div_up(n, 256), 256, 0, stream>>>(n, pos_view, deep);
}

void pose_contain_target(const float4* out_rgba, int W, int H, int r, float thr, uint32_t* rgba, uint8_t* weights, cudaStream_t stream) {
  dim3 b(16, 16), g((W + 15) / 16, (H + 15) / 16);
  target_kernel<<<g, b, 0, stream>>>(out_rgba, W, H, r, thr, rgba, weights);
}

}  // namespace b2c
