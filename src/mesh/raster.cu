#include "mesh/raster.h"
#include "util/log.h"
#include "mesh/geom.cuh"
#include <algorithm>
#include <unordered_map>
#include <cmath>

namespace b2c {

CameraGPU scaled_camera(const CameraGPU& c, int ss) {
  CameraGPU s = c; s.fx *= ss; s.fy *= ss; s.cx *= ss; s.cy *= ss; s.W *= ss; s.H *= ss; return s;
}

namespace {

constexpr unsigned long long INF_KEY = ~0ull;

__global__ void clear_keys(size_t n, unsigned long long* k) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) k[i] = INF_KEY; }

__device__ __forceinline__ float edge_fn(float ax, float ay, float bx, float by, float px, float py) { return (bx - ax) * (py - ay) - (by - ay) * (px - ax); }

// One thread per triangle: project, scan the clamped screen bbox, atomicMin a (depth bits, id) key per covered pixel.
__global__ void tri_kernel(int nf, const float3* __restrict__ v, const uint3* __restrict__ f, CameraGPU cam, unsigned long long* __restrict__ keys) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nf) return;
  uint3 fc = f[t]; float3 p[3] = {v[fc.x], v[fc.y], v[fc.z]};
  float sx[3], sy[3], z[3];
  for (int k = 0; k < 3; k++) { z[k] = project_point(cam, p[k].x, p[k].y, p[k].z, sx[k], sy[k]); if (!(z[k] > 1e-4f)) return; }
  float area = edge_fn(sx[0], sy[0], sx[1], sy[1], sx[2], sy[2]);
  if (fabsf(area) < 1e-12f) return;
  float inv = 1.f / area;
  int x0 = max((int)ceilf(fminf(sx[0], fminf(sx[1], sx[2])) - 0.5f), 0), x1 = min((int)floorf(fmaxf(sx[0], fmaxf(sx[1], sx[2])) - 0.5f), cam.W - 1);
  int y0 = max((int)ceilf(fminf(sy[0], fminf(sy[1], sy[2])) - 0.5f), 0), y1 = min((int)floorf(fmaxf(sy[0], fmaxf(sy[1], sy[2])) - 0.5f), cam.H - 1);
  if (x1 < x0 || y1 < y0) return;
  float iz0 = 1.f / z[0], iz1 = 1.f / z[1], iz2 = 1.f / z[2];
  for (int y = y0; y <= y1; y++) {
    float py = y + 0.5f;
    for (int x = x0; x <= x1; x++) {
      float px = x + 0.5f;
      float b0 = edge_fn(sx[1], sy[1], sx[2], sy[2], px, py) * inv;
      float b1 = edge_fn(sx[2], sy[2], sx[0], sy[0], px, py) * inv;
      float b2 = edge_fn(sx[0], sy[0], sx[1], sy[1], px, py) * inv;
      if (b0 < -1e-6f || b1 < -1e-6f || b2 < -1e-6f) continue;
      float zp = 1.f / (b0 * iz0 + b1 * iz1 + b2 * iz2);
      unsigned long long key = ((unsigned long long)__float_as_uint(zp) << 32) | (unsigned)t;
      atomicMin(keys + (size_t)y * cam.W + x, key);
    }
  }
}

__global__ void resolve_kernel(size_t n, const unsigned long long* __restrict__ keys, const float3* __restrict__ v, const uint3* __restrict__ f, CameraGPU cam, bool with_bary,
                               float* __restrict__ z, int* __restrict__ tri, float2* __restrict__ bary) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long long k = keys[i];
  if (k == INF_KEY) { z[i] = 0.f; if (with_bary) { tri[i] = -1; bary[i] = make_float2(0.f, 0.f); } return; }
  z[i] = __uint_as_float((unsigned)(k >> 32));
  if (!with_bary) return;
  int t = (int)(k & 0xffffffffu); tri[i] = t;
  uint3 fc = f[t]; float3 p[3] = {v[fc.x], v[fc.y], v[fc.z]};
  float sx[3], sy[3], zz[3];
  for (int c = 0; c < 3; c++) zz[c] = project_point(cam, p[c].x, p[c].y, p[c].z, sx[c], sy[c]);
  float px = (float)(i % cam.W) + 0.5f, py = (float)(i / cam.W) + 0.5f;
  float area = edge_fn(sx[0], sy[0], sx[1], sy[1], sx[2], sy[2]), inv = 1.f / area;
  float b0 = edge_fn(sx[1], sy[1], sx[2], sy[2], px, py) * inv, b1 = edge_fn(sx[2], sy[2], sx[0], sy[0], px, py) * inv, b2 = 1.f - b0 - b1;
  float w0 = b0 / zz[0], w1 = b1 / zz[1], w2 = b2 / zz[2], s = w0 + w1 + w2;
  bary[i] = make_float2(w1 / s, w2 / s);
}

__global__ void face_normal_kernel(int nf, const float3* __restrict__ v, const uint3* __restrict__ f, float3* __restrict__ out) {
  int t = blockIdx.x * blockDim.x + threadIdx.x; if (t >= nf) return;
  uint3 fc = f[t]; float3 a = v[fc.x], b = v[fc.y], c = v[fc.z];
  float3 e1 = b - a, e2 = c - a;
  float3 n = make_float3(e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x);
  float l = len3(n); out[t] = l > 1e-30f ? n * (1.f / l) : make_float3(0.f, 0.f, 0.f);
}
__global__ void accum_normal_kernel(int nf, const float3* __restrict__ v, const uint3* __restrict__ f, float3* __restrict__ acc) {
  int t = blockIdx.x * blockDim.x + threadIdx.x; if (t >= nf) return;
  uint3 fc = f[t]; float3 a = v[fc.x], b = v[fc.y], c = v[fc.z];
  float3 e1 = b - a, e2 = c - a;
  float3 n = make_float3(e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x);
  unsigned idx[3] = {fc.x, fc.y, fc.z};
  for (int k = 0; k < 3; k++) { atomicAdd(&acc[idx[k]].x, n.x); atomicAdd(&acc[idx[k]].y, n.y); atomicAdd(&acc[idx[k]].z, n.z); }
}
__global__ void normalise_kernel(int n, float3* __restrict__ a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float3 v = a[i]; float l = len3(v); a[i] = l > 1e-12f ? v * (1.f / l) : make_float3(0.f, 0.f, 0.f);
}

__global__ void closest_kernel(TriGridDev g, const float3* __restrict__ q, int n, int max_rings, int* __restrict__ tri_out, float3* __restrict__ pt_out, float* __restrict__ dist_out, float3* __restrict__ bary_out, float* __restrict__ sign_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float3 p = q[i];
  int cx = min(max((int)floorf((p.x - g.lo.x) / g.cell), 0), g.dims.x - 1);
  int cy = min(max((int)floorf((p.y - g.lo.y) / g.cell), 0), g.dims.y - 1);
  int cz = min(max((int)floorf((p.z - g.lo.z) / g.cell), 0), g.dims.z - 1);
  float best = INFINITY; int best_t = -1; float3 best_p = p, best_b = make_float3(0, 0, 0); int best_r = 0;
  for (int r = 0; r <= max_rings; r++) {
    if (r > 0 && best <= (float)(r - 1) * g.cell) break;  // rings < r are done: every unexamined point is >= (r - 1) cells away
    int x0 = cx - r, x1 = cx + r, y0 = cy - r, y1 = cy + r, z0 = cz - r, z1 = cz + r;
    if (x0 < 0 && y0 < 0 && z0 < 0 && x1 >= g.dims.x && y1 >= g.dims.y && z1 >= g.dims.z && r > 0) break;  // ring beyond the grid
    for (int x = max(x0, 0); x <= min(x1, g.dims.x - 1); x++)
      for (int y = max(y0, 0); y <= min(y1, g.dims.y - 1); y++)
        for (int z = max(z0, 0); z <= min(z1, g.dims.z - 1); z++) {
          bool shell = (x == x0 || x == x1 || y == y0 || y == y1 || z == z0 || z == z1);
          if (!shell) continue;
          int c = (x * g.dims.y + y) * g.dims.z + z;
          for (int k = g.cell_start[c]; k < g.cell_start[c + 1]; k++) {
            int t = g.items[k]; uint3 fc = g.f[t];
            float3 b; int reg; float3 cp = closest_on_tri(p, g.v[fc.x], g.v[fc.y], g.v[fc.z], b, reg);
            float d = len3(cp - p);
            if (d < best || (d == best && t < best_t)) { best = d; best_t = t; best_p = cp; best_b = b; best_r = reg; }
          }
        }
  }
  tri_out[i] = best_t; pt_out[i] = best_p; dist_out[i] = best_t >= 0 ? best : INFINITY;
  if (bary_out) bary_out[i] = best_b;
  if (sign_out) {
    float s = 1.f;
    if (best_t >= 0 && g.fn) {
      uint3 fc = g.f[best_t]; float3 nrm;
      if (best_r == 0) nrm = g.fn[best_t];
      else if (best_r <= 3) { unsigned vi = best_r == 1 ? fc.x : best_r == 2 ? fc.y : fc.z; nrm = g.vn[vi]; }
      else nrm = g.en[best_t * 3 + (best_r - 4)];
      s = dot3(p - best_p, nrm) >= 0.f ? 1.f : -1.f;
    }
    sign_out[i] = s;
  }
}

__global__ void point_query_kernel(PointGridDev g, const float3* __restrict__ q, int n, float radius, float* __restrict__ dist_out, int* __restrict__ count_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float3 p = q[i]; int R = (int)ceilf(radius / g.cell); float r2 = radius * radius;
  int cx = (int)floorf((p.x - g.lo.x) / g.cell), cy = (int)floorf((p.y - g.lo.y) / g.cell), cz = (int)floorf((p.z - g.lo.z) / g.cell);
  float best = INFINITY; int cnt = 0;
  for (int x = max(cx - R, 0); x <= min(cx + R, g.dims.x - 1); x++)
    for (int y = max(cy - R, 0); y <= min(cy + R, g.dims.y - 1); y++)
      for (int z = max(cz - R, 0); z <= min(cz + R, g.dims.z - 1); z++) {
        int c = (x * g.dims.y + y) * g.dims.z + z;
        for (int k = g.cell_start[c]; k < g.cell_start[c + 1]; k++) {
          float3 d = g.p[g.items[k]] - p; float d2 = dot3(d, d);
          if (d2 <= r2) { cnt++; best = fminf(best, d2); }
        }
      }
  if (dist_out) dist_out[i] = sqrtf(best);
  if (count_out) count_out[i] = cnt;
}

}  // namespace

void MeshRaster::upload(const TriMesh& m, cudaStream_t stream) {
  nv = (int)m.nv(); nf = (int)m.nf();
  std::vector<float3> hv(nv); for (int i = 0; i < nv; i++) hv[i] = make_float3(m.v[i * 3], m.v[i * 3 + 1], m.v[i * 3 + 2]);
  std::vector<uint3> hf(nf); for (int i = 0; i < nf; i++) hf[i] = make_uint3(m.f[i * 3], m.f[i * 3 + 1], m.f[i * 3 + 2]);
  v.upload(hv, stream); f.upload(hf, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}
void MeshRaster::set_vertices(const float3* dev_v, cudaStream_t stream) {
  CUDA_CHECK(cudaMemcpyAsync(v.ptr, dev_v, (size_t)nv * sizeof(float3), cudaMemcpyDeviceToDevice, stream));
}
void MeshRaster::raster(const CameraGPU& cam0, int ss, bool with_bary, cudaStream_t stream) {
  CameraGPU cam = scaled_camera(cam0, ss);
  W = cam.W; H = cam.H; size_t n = (size_t)W * H;
  keys.reserve(n); z.reserve(n);
  if (with_bary) { tri.reserve(n); bary.reserve(n); }
  clear_keys<<<div_up(n, 256), 256, 0, stream>>>(n, keys); CUDA_KERNEL_CHECK();
  if (nf > 0) { tri_kernel<<<div_up(nf, 128), 128, 0, stream>>>(nf, v, f, cam, keys); CUDA_KERNEL_CHECK(); }
  resolve_kernel<<<div_up(n, 256), 256, 0, stream>>>(n, keys, v, f, cam, with_bary, z, with_bary ? tri.ptr : nullptr, with_bary ? bary.ptr : nullptr);
  CUDA_KERNEL_CHECK();
}

void vertex_normals_gpu(const float3* v, int nv, const uint3* f, int nf, float3* out, cudaStream_t stream) {
  CUDA_CHECK(cudaMemsetAsync(out, 0, (size_t)nv * sizeof(float3), stream));
  accum_normal_kernel<<<div_up(nf, 256), 256, 0, stream>>>(nf, v, f, out); CUDA_KERNEL_CHECK();
  normalise_kernel<<<div_up(nv, 256), 256, 0, stream>>>(nv, out); CUDA_KERNEL_CHECK();
}
void face_normals_gpu(const float3* v, const uint3* f, int nf, float3* out, cudaStream_t stream) {
  face_normal_kernel<<<div_up(nf, 256), 256, 0, stream>>>(nf, v, f, out); CUDA_KERNEL_CHECK();
}

void Adjacency::build(const TriMesh& m) {
  size_t nv = m.nv(), nf = m.nf();
  std::vector<uint64_t> e; e.reserve(nf * 6);
  for (size_t t = 0; t < nf; t++) {
    uint32_t a = m.f[t * 3], b = m.f[t * 3 + 1], c = m.f[t * 3 + 2];
    uint32_t p[3][2] = {{a, b}, {b, c}, {c, a}};
    for (auto& q : p) { e.push_back(((uint64_t)q[0] << 32) | q[1]); e.push_back(((uint64_t)q[1] << 32) | q[0]); }
  }
  std::sort(e.begin(), e.end()); e.erase(std::unique(e.begin(), e.end()), e.end());
  start.assign(nv + 1, 0); nbr.resize(e.size());
  for (size_t i = 0; i < e.size(); i++) { start[(e[i] >> 32) + 1]++; nbr[i] = (int)(e[i] & 0xffffffffu); }
  for (size_t i = 0; i < nv; i++) start[i + 1] += start[i];
}
void Adjacency::upload(cudaStream_t stream) { d_start.upload(start, stream); d_nbr.upload(nbr, stream); CUDA_CHECK(cudaStreamSynchronize(stream)); }
int Adjacency::max_degree() const { int d = 0; for (size_t i = 0; i + 1 < start.size(); i++) d = std::max(d, start[i + 1] - start[i]); return d; }

namespace {
template <typename Fill>
void build_cells(int3 dims, size_t nitems, const Fill& fill, std::vector<int>& cell_start, std::vector<int>& items) {
  // fill(i, push) calls push(cell) for every cell item i touches.
  std::vector<std::pair<int, int>> pairs; pairs.reserve(nitems * 4);
  for (size_t i = 0; i < nitems; i++) fill(i, [&](int c) { pairs.emplace_back(c, (int)i); });
  std::sort(pairs.begin(), pairs.end());
  size_t nc = (size_t)dims.x * dims.y * dims.z;
  cell_start.assign(nc + 1, 0); items.resize(pairs.size());
  for (size_t i = 0; i < pairs.size(); i++) { cell_start[pairs[i].first + 1]++; items[i] = pairs[i].second; }
  for (size_t c = 0; c < nc; c++) cell_start[c + 1] += cell_start[c];
}
}  // namespace

void TriGrid::build(const TriMesh& m, float cell_size, bool with_normals, cudaStream_t stream) {
  nv = (int)m.nv(); nf = (int)m.nf(); cell = cell_size;
  float lo3[3] = {INFINITY, INFINITY, INFINITY}, hi3[3] = {-INFINITY, -INFINITY, -INFINITY};
  for (size_t i = 0; i < m.nv(); i++) for (int k = 0; k < 3; k++) { lo3[k] = std::min(lo3[k], m.v[i * 3 + k]); hi3[k] = std::max(hi3[k], m.v[i * 3 + k]); }
  lo = make_float3(lo3[0] - cell, lo3[1] - cell, lo3[2] - cell); hi = make_float3(hi3[0] + cell, hi3[1] + cell, hi3[2] + cell);
  dims = make_int3((int)std::ceil((hi.x - lo.x) / cell) + 1, (int)std::ceil((hi.y - lo.y) / cell) + 1, (int)std::ceil((hi.z - lo.z) / cell) + 1);
  std::vector<int> cs, it;
  build_cells(dims, m.nf(), [&](size_t t, auto push) {
    int c0[3] = {1 << 30, 1 << 30, 1 << 30}, c1[3] = {-1, -1, -1};
    for (int k = 0; k < 3; k++) { const float* p = &m.v[m.f[t * 3 + k] * 3]; float pc[3] = {p[0], p[1], p[2]}; float lo_[3] = {lo.x, lo.y, lo.z};
      for (int a = 0; a < 3; a++) { int c = (int)std::floor((pc[a] - lo_[a]) / cell); c0[a] = std::min(c0[a], c); c1[a] = std::max(c1[a], c); } }
    for (int x = std::max(c0[0], 0); x <= std::min(c1[0], dims.x - 1); x++) for (int y = std::max(c0[1], 0); y <= std::min(c1[1], dims.y - 1); y++) for (int z = std::max(c0[2], 0); z <= std::min(c1[2], dims.z - 1); z++) push((x * dims.y + y) * dims.z + z);
  }, cs, it);
  cell_start.upload(cs, stream); items.upload(it, stream);
  std::vector<float3> hv(nv); for (int i = 0; i < nv; i++) hv[i] = make_float3(m.v[i * 3], m.v[i * 3 + 1], m.v[i * 3 + 2]);
  std::vector<uint3> hf(nf); for (int i = 0; i < nf; i++) hf[i] = make_uint3(m.f[i * 3], m.f[i * 3 + 1], m.f[i * 3 + 2]);
  v.upload(hv, stream); f.upload(hf, stream);
  if (with_normals) {
    // Pseudo-normals: face normals, angle-weighted vertex normals, edge normals = the sum of the two adjacent face normals.
    std::vector<float3> hfn(nf), hvn(nv, make_float3(0, 0, 0)), hen(nf * 3, make_float3(0, 0, 0));
    std::unordered_map<uint64_t, float3> edge_acc; edge_acc.reserve(nf * 2);
    auto key = [](uint32_t a, uint32_t b) { return a < b ? ((uint64_t)a << 32 | b) : ((uint64_t)b << 32 | a); };
    auto norm = [](float3 a) { float l = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z); return l > 1e-30f ? make_float3(a.x / l, a.y / l, a.z / l) : make_float3(0, 0, 0); };
    auto sub = [](float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); };
    auto cross = [](float3 a, float3 b) { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); };
    auto dot = [](float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; };
    for (int t = 0; t < nf; t++) {
      uint3 fc = hf[t]; float3 p[3] = {hv[fc.x], hv[fc.y], hv[fc.z]};
      float3 n = norm(cross(sub(p[1], p[0]), sub(p[2], p[0]))); hfn[t] = n;
      unsigned idx[3] = {fc.x, fc.y, fc.z};
      for (int k = 0; k < 3; k++) {
        float3 e1 = norm(sub(p[(k + 1) % 3], p[k])), e2 = norm(sub(p[(k + 2) % 3], p[k]));
        float ang = std::acos(std::min(1.f, std::max(-1.f, dot(e1, e2))));
        hvn[idx[k]].x += n.x * ang; hvn[idx[k]].y += n.y * ang; hvn[idx[k]].z += n.z * ang;
        auto& acc = edge_acc[key(idx[k], idx[(k + 1) % 3])]; acc.x += n.x; acc.y += n.y; acc.z += n.z;
      }
    }
    for (auto& x : hvn) x = norm(x);
    for (int t = 0; t < nf; t++) { uint3 fc = hf[t]; unsigned idx[3] = {fc.x, fc.y, fc.z}; for (int k = 0; k < 3; k++) hen[t * 3 + k] = norm(edge_acc[key(idx[k], idx[(k + 1) % 3])]); }
    fn.upload(hfn, stream); vn.upload(hvn, stream); en.upload(hen, stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
}
TriGridDev TriGrid::dev() const {
  TriGridDev d; d.lo = lo; d.cell = cell; d.dims = dims; d.cell_start = cell_start; d.items = items; d.v = v; d.f = f; d.fn = fn.ptr; d.vn = vn.ptr; d.en = en.ptr; d.nf = nf; return d;
}
void closest_points_gpu(const TriGridDev& g, const float3* q, int n, int max_rings, int* tri, float3* point, float* dist, float3* bary, float* sign, cudaStream_t stream) {
  if (n <= 0) return;
  closest_kernel<<<div_up(n, 128), 128, 0, stream>>>(g, q, n, max_rings, tri, point, dist, bary, sign); CUDA_KERNEL_CHECK();
}

void PointGrid::build(const std::vector<float>& xyz, float cell_size, cudaStream_t stream) {
  n = (int)(xyz.size() / 3); cell = cell_size;
  float lo3[3] = {INFINITY, INFINITY, INFINITY}, hi3[3] = {-INFINITY, -INFINITY, -INFINITY};
  for (int i = 0; i < n; i++) for (int k = 0; k < 3; k++) { lo3[k] = std::min(lo3[k], xyz[i * 3 + k]); hi3[k] = std::max(hi3[k], xyz[i * 3 + k]); }
  if (n == 0) { lo3[0] = lo3[1] = lo3[2] = 0; hi3[0] = hi3[1] = hi3[2] = 0; }
  lo = make_float3(lo3[0] - cell, lo3[1] - cell, lo3[2] - cell); hi = make_float3(hi3[0] + cell, hi3[1] + cell, hi3[2] + cell);
  dims = make_int3((int)std::ceil((hi.x - lo.x) / cell) + 1, (int)std::ceil((hi.y - lo.y) / cell) + 1, (int)std::ceil((hi.z - lo.z) / cell) + 1);
  std::vector<int> cs, it;
  build_cells(dims, (size_t)n, [&](size_t i, auto push) {
    int x = (int)std::floor((xyz[i * 3] - lo.x) / cell), y = (int)std::floor((xyz[i * 3 + 1] - lo.y) / cell), z = (int)std::floor((xyz[i * 3 + 2] - lo.z) / cell);
    push((x * dims.y + y) * dims.z + z);
  }, cs, it);
  cell_start.upload(cs, stream); items.upload(it, stream);
  std::vector<float3> hp(n); for (int i = 0; i < n; i++) hp[i] = make_float3(xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2]);
  p.upload(hp, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}
PointGridDev PointGrid::dev() const { PointGridDev d; d.lo = lo; d.cell = cell; d.dims = dims; d.cell_start = cell_start; d.items = items; d.p = p; d.n = n; return d; }
void point_query_gpu(const PointGridDev& g, const float3* q, int n, float radius, float* dist, int* count, cudaStream_t stream) {
  if (n <= 0) return;
  point_query_kernel<<<div_up(n, 128), 128, 0, stream>>>(g, q, n, radius, dist, count); CUDA_KERNEL_CHECK();
}

}  // namespace b2c
