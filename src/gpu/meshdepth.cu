#include "gpu/meshdepth.h"
#include "gpu/splat_math.cuh"

namespace b2c {

namespace {

constexpr uint32_t INF_BITS = 0x7f800000u;

__global__ void clear_kernel(int n, uint32_t* __restrict__ z) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) z[i] = INF_BITS; }

// One thread per triangle: project, then scan the clamped bounding box with edge functions; depth is
// perspective-correct (1/z interpolated in screen space). Nearest depth wins through atomicMin on the float bits
// (positive floats order like their bit patterns).
__global__ void tri_kernel(int nf, const float3* __restrict__ verts, const uint3* __restrict__ faces, CamDev cam, int W, int H, uint32_t* __restrict__ zbuf) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nf) return;
  uint3 fc = faces[t];
  float3 p[3] = {verts[fc.x], verts[fc.y], verts[fc.z]};
  float sx[3], sy[3], iz[3];
  for (int k = 0; k < 3; k++) {
    float x = cam.R[0] * p[k].x + cam.R[1] * p[k].y + cam.R[2] * p[k].z + cam.t[0];
    float y = cam.R[3] * p[k].x + cam.R[4] * p[k].y + cam.R[5] * p[k].z + cam.t[1];
    float z = cam.R[6] * p[k].x + cam.R[7] * p[k].y + cam.R[8] * p[k].z + cam.t[2];
    if (!(z > 1e-3f)) return;  // behind or on the camera plane: skip the triangle (the body is never there)
    sx[k] = cam.fx * x / z + cam.cx; sy[k] = cam.fy * y / z + cam.cy; iz[k] = 1.f / z;
  }
  float area = (sx[1] - sx[0]) * (sy[2] - sy[0]) - (sx[2] - sx[0]) * (sy[1] - sy[0]);
  if (fabsf(area) < 1e-12f) return;
  float inv_area = 1.f / area;
  int x0 = max((int)floorf(fminf(sx[0], fminf(sx[1], sx[2])) - 0.5f), 0), x1 = min((int)ceilf(fmaxf(sx[0], fmaxf(sx[1], sx[2])) - 0.5f), W - 1);
  int y0 = max((int)floorf(fminf(sy[0], fminf(sy[1], sy[2])) - 0.5f), 0), y1 = min((int)ceilf(fmaxf(sy[0], fmaxf(sy[1], sy[2])) - 0.5f), H - 1);
  for (int y = y0; y <= y1; y++) {
    float py = y + 0.5f;
    for (int x = x0; x <= x1; x++) {
      float px = x + 0.5f;
      float w0 = ((sx[1] - px) * (sy[2] - py) - (sx[2] - px) * (sy[1] - py)) * inv_area;
      float w1 = ((sx[2] - px) * (sy[0] - py) - (sx[0] - px) * (sy[2] - py)) * inv_area;
      float w2 = 1.f - w0 - w1;
      if (w0 < 0.f || w1 < 0.f || w2 < 0.f) continue;
      float z = 1.f / (w0 * iz[0] + w1 * iz[1] + w2 * iz[2]);
      atomicMin(zbuf + y * W + x, __float_as_uint(z));
    }
  }
}

__global__ void dilate_kernel(int W, int H, int r, const uint32_t* __restrict__ zbuf, float* __restrict__ out) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  uint32_t best = 0u;
  for (int dy = -r; dy <= r; dy++) {
    int yy = y + dy; if (yy < 0 || yy >= H) { best = INF_BITS; continue; }  // outside the frame counts as not hit
    for (int dx = -r; dx <= r; dx++) {
      int xx = x + dx; if (xx < 0 || xx >= W) { best = INF_BITS; continue; }
      best = max(best, zbuf[yy * W + xx]);
    }
  }
  out[y * W + x] = __uint_as_float(best);
}

}  // namespace

void MeshGPU::upload(const TriMesh& m, cudaStream_t stream) {
  nv = (int)m.nv(); nf = (int)m.nf();
  std::vector<float3> hv(nv); for (int i = 0; i < nv; i++) hv[i] = make_float3(m.v[i * 3], m.v[i * 3 + 1], m.v[i * 3 + 2]);
  std::vector<uint3> hf(nf); for (int i = 0; i < nf; i++) hf[i] = make_uint3(m.f[i * 3], m.f[i * 3 + 1], m.f[i * 3 + 2]);
  v.upload(hv, stream); f.upload(hf, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

CamDev to_camdev(const CameraGPU& c);

void MeshGPU::rasterize(const CameraGPU& cam, int W, int H, int dilate, cudaStream_t stream) {
  size_t npx = (size_t)W * H;
  zbuf.reserve(npx); depth.reserve(npx);
  clear_kernel<<<div_up((int)npx, 256), 256, 0, stream>>>((int)npx, zbuf);
  CUDA_KERNEL_CHECK();
  if (nf > 0) {
    tri_kernel<<<div_up(nf, 128), 128, 0, stream>>>(nf, v, f, to_camdev(cam), W, H, zbuf);
    CUDA_KERNEL_CHECK();
  }
  dim3 block(16, 16), grid((W + 15) / 16, (H + 15) / 16);
  dilate_kernel<<<grid, block, 0, stream>>>(W, H, max(dilate, 0), zbuf, depth);
  CUDA_KERNEL_CHECK();
}

}  // namespace b2c
