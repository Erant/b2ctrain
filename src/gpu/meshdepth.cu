#include "gpu/meshdepth.h"
#include "gpu/splat_math.cuh"
#include <thread>
#include <atomic>
#include <algorithm>
#include <cmath>

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

// One thread per surfel: a disc of radius p.w in the point's tangent plane. Each pixel ray in the disc's screen
// bounding box is intersected with the plane; hits within the radius write their depth.
__global__ void surfel_kernel(int np, const float4* __restrict__ pts, const float4* __restrict__ nrm, CamDev cam, int W, int H, uint32_t* __restrict__ zbuf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= np) return;
  float4 p = pts[i], nw = nrm[i];
  // Camera space: centre and normal.
  float px = cam.R[0] * p.x + cam.R[1] * p.y + cam.R[2] * p.z + cam.t[0];
  float py = cam.R[3] * p.x + cam.R[4] * p.y + cam.R[5] * p.z + cam.t[1];
  float pz = cam.R[6] * p.x + cam.R[7] * p.y + cam.R[8] * p.z + cam.t[2];
  if (!(pz > 1e-3f)) return;
  float nx = cam.R[0] * nw.x + cam.R[1] * nw.y + cam.R[2] * nw.z;
  float ny = cam.R[3] * nw.x + cam.R[4] * nw.y + cam.R[5] * nw.z;
  float nz = cam.R[6] * nw.x + cam.R[7] * nw.y + cam.R[8] * nw.z;
  float r = p.w, r2 = r * r;
  // Conservative screen bbox: the disc lies within the sphere of radius r.
  float zn = fmaxf(pz - r, 1e-3f);
  float sx = cam.fx * px / pz + cam.cx, sy = cam.fy * py / pz + cam.cy;
  float rx = r * cam.fx / zn, ry = r * cam.fy / zn;
  int x0 = max((int)floorf(sx - rx - 0.5f), 0), x1 = min((int)ceilf(sx + rx - 0.5f), W - 1);
  int y0 = max((int)floorf(sy - ry - 0.5f), 0), y1 = min((int)ceilf(sy + ry - 0.5f), H - 1);
  if ((x1 - x0 + 1) * (y1 - y0 + 1) > 65536) return;  // a surfel almost touching the camera: not a body surface
  float ndp = nx * px + ny * py + nz * pz;
  for (int yy = y0; yy <= y1; yy++) {
    float dy = (yy + 0.5f - cam.cy) / cam.fy;
    for (int xx = x0; xx <= x1; xx++) {
      float dx = (xx + 0.5f - cam.cx) / cam.fx;
      float denom = nx * dx + ny * dy + nz;
      if (fabsf(denom) < 1e-5f) continue;
      float t = ndp / denom;  // ray (dx, dy, 1) * t hits the plane
      if (!(t > 1e-3f)) continue;
      float qx = dx * t - px, qy = dy * t - py, qz = t - pz;
      if (qx * qx + qy * qy + qz * qz > r2) continue;
      atomicMin(zbuf + yy * W + xx, __float_as_uint(t));
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

void MeshGPU::upload_points(const float* xyz, size_t n, float radius, cudaStream_t stream) {
  np = (int)n;
  std::vector<float4> h(n), hn(n);
  for (size_t i = 0; i < n; i++) h[i] = make_float4(xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2], radius);
  // PCA normal per point over its K nearest neighbours (smallest-eigenvalue direction of the covariance).
  constexpr int K = 12;
  unsigned nt = std::max(1u, std::min(std::thread::hardware_concurrency(), 32u));
  std::atomic<size_t> next{0};
  auto worker = [&]() {
    std::vector<std::pair<float, size_t>> best;
    for (size_t i; (i = next.fetch_add(64)) < n;) {
      size_t end = std::min(n, i + 64);
      for (size_t p = i; p < end; p++) {
        float px = xyz[p * 3], py = xyz[p * 3 + 1], pz = xyz[p * 3 + 2];
        best.clear();
        float worst = INFINITY;
        for (size_t q = 0; q < n; q++) {
          if (q == p) continue;
          float dx = xyz[q * 3] - px, dy = xyz[q * 3 + 1] - py, dz = xyz[q * 3 + 2] - pz;
          float d = dx * dx + dy * dy + dz * dz;
          if (best.size() < K) { best.emplace_back(d, q); if (best.size() == K) { std::sort(best.begin(), best.end()); worst = best.back().first; } }
          else if (d < worst) { best.back() = {d, q}; std::sort(best.begin(), best.end()); worst = best.back().first; }
        }
        double mx = px, my = py, mz = pz; int cnt = 1;
        for (auto& [d, q] : best) { mx += xyz[q * 3]; my += xyz[q * 3 + 1]; mz += xyz[q * 3 + 2]; cnt++; }
        mx /= cnt; my /= cnt; mz /= cnt;
        double c[6] = {0, 0, 0, 0, 0, 0};  // xx xy xz yy yz zz
        auto acc = [&](double x, double y, double z) { x -= mx; y -= my; z -= mz; c[0] += x * x; c[1] += x * y; c[2] += x * z; c[3] += y * y; c[4] += y * z; c[5] += z * z; };
        acc(px, py, pz); for (auto& [d, q] : best) acc(xyz[q * 3], xyz[q * 3 + 1], xyz[q * 3 + 2]);
        // Smallest eigenvector by inverse iteration on (C + eps I)^-1 (3x3, a few steps from a fixed start).
        double tr = c[0] + c[3] + c[5]; double eps = 1e-9 * std::max(tr, 1e-30);
        double A[9] = {c[0] + eps, c[1], c[2], c[1], c[3] + eps, c[4], c[2], c[4], c[5] + eps};
        double det = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
        double v[3] = {0.577, 0.577, 0.577};
        if (std::abs(det) > 1e-300) {
          double inv[9] = {(A[4] * A[8] - A[5] * A[7]) / det, (A[2] * A[7] - A[1] * A[8]) / det, (A[1] * A[5] - A[2] * A[4]) / det,
                           (A[5] * A[6] - A[3] * A[8]) / det, (A[0] * A[8] - A[2] * A[6]) / det, (A[2] * A[3] - A[0] * A[5]) / det,
                           (A[3] * A[7] - A[4] * A[6]) / det, (A[1] * A[6] - A[0] * A[7]) / det, (A[0] * A[4] - A[1] * A[3]) / det};
          for (int it = 0; it < 12; it++) {
            double w[3] = {inv[0] * v[0] + inv[1] * v[1] + inv[2] * v[2], inv[3] * v[0] + inv[4] * v[1] + inv[5] * v[2], inv[6] * v[0] + inv[7] * v[1] + inv[8] * v[2]};
            double l = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]); if (l < 1e-300) break;
            v[0] = w[0] / l; v[1] = w[1] / l; v[2] = w[2] / l;
          }
        }
        hn[p] = make_float4((float)v[0], (float)v[1], (float)v[2], 0.f);
      }
    }
  };
  std::vector<std::thread> threads; for (unsigned t = 0; t < nt; t++) threads.emplace_back(worker); for (auto& t : threads) t.join();
  pts.upload(h, stream); nrm.upload(hn, stream);
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
  if (np > 0) {
    surfel_kernel<<<div_up(np, 128), 128, 0, stream>>>(np, pts, nrm, to_camdev(cam), W, H, zbuf);
    CUDA_KERNEL_CHECK();
  }
  dim3 block(16, 16), grid((W + 15) / 16, (H + 15) / 16);
  dilate_kernel<<<grid, block, 0, stream>>>(W, H, max(dilate, 0), zbuf, depth);
  CUDA_KERNEL_CHECK();
}

}  // namespace b2c
