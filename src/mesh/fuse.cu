// `b2ctrain mesh-fuse`: see fuse.h. Reference: out/mesh/tools/tsdf_torch.py (+ body_sdf.py for the prior).
#include "mesh/fuse.h"
#include "mesh/common.h"
#include "mesh/raster.h"
#include "mesh/geom.cuh"
#include "util/log.h"
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>

namespace b2c {

VolumeGrid VolumeGrid::from_bbox(const float* lo, const float* hi, float voxel) {
  VolumeGrid g; g.voxel = voxel; g.lo = make_float3(lo[0], lo[1], lo[2]);
  g.dims = make_int3((int)std::ceil((hi[0] - lo[0]) / voxel) + 1, (int)std::ceil((hi[1] - lo[1]) / voxel) + 1, (int)std::ceil((hi[2] - lo[2]) / voxel) + 1);
  return g;
}

namespace {

#include "mesh/mc_tables.inc"

constexpr int MAX_BATCH = 32;
struct ViewDev { CameraGPU cam; const float* depth; float weight; int allowed; };
struct CarveDev { CameraGPU cam; const uint8_t* mask; };
__constant__ ViewDev c_views[MAX_BATCH];
__constant__ CarveDev c_carve[MAX_BATCH];

__device__ __forceinline__ float3 grid_point(const VolumeGrid& g, size_t i, int& x, int& y, int& z) {
  z = (int)(i % g.dims.z); size_t r = i / g.dims.z; y = (int)(r % g.dims.y); x = (int)(r / g.dims.y);
  return make_float3(g.lo.x + x * g.voxel, g.lo.y + y * g.voxel, g.lo.z + z * g.voxel);
}

__global__ void integrate_kernel(VolumeGrid g, float* __restrict__ sdf, float* __restrict__ wgt, const uint8_t* __restrict__ protect, int nviews, float trunc) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= g.n()) return;
  int x, y, z; float3 p = grid_point(g, i, x, y, z);
  float s = sdf[i], w = wgt[i]; bool prot = protect && protect[i];
  for (int k = 0; k < nviews; k++) {
    const ViewDev& v = c_views[k];
    if (prot && !v.allowed) continue;
    float u, vv; float zc = project_point(v.cam, p.x, p.y, p.z, u, vv);
    if (!(zc > 0.f)) continue;
    int ui = (int)floorf(u), vi = (int)floorf(vv);
    if (ui < 0 || vi < 0 || ui >= v.cam.W || vi >= v.cam.H) continue;
    float d = v.depth[(size_t)vi * v.cam.W + ui];
    if (!(d > 0.f)) continue;
    float sd = d - zc;
    if (!(sd > -trunc)) continue;
    sd = fminf(fmaxf(sd / trunc, -1.f), 1.f);
    float wn = w + v.weight;
    s = (s * w + v.weight * sd) / fmaxf(wn, 1e-6f); w = wn;
  }
  sdf[i] = s; wgt[i] = w;
}

__global__ void carve_kernel(VolumeGrid g, uint8_t* __restrict__ carved, int nviews) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= g.n()) return;
  int x, y, z; float3 p = grid_point(g, i, x, y, z);
  int c = carved[i];
  for (int k = 0; k < nviews; k++) {
    const CarveDev& v = c_carve[k];
    float u, vv; float zc = project_point(v.cam, p.x, p.y, p.z, u, vv);
    if (!(zc > 0.f)) continue;
    int ui = (int)floorf(u), vi = (int)floorf(vv);
    if (ui < 0 || vi < 0 || ui >= v.cam.W || vi >= v.cam.H) continue;
    if (v.mask[(size_t)vi * v.cam.W + ui] == 0) c++;
  }
  carved[i] = (uint8_t)min(c, 255);
}

// ---- per-view preprocessing (the reference's cv2 steps) ----
__global__ void depth_prep_kernel(int W, int H, const uint16_t* __restrict__ d16, const uint8_t* __restrict__ alpha, int alpha_min, float* __restrict__ d, uint8_t* __restrict__ valid) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y; if (x >= W || y >= H) return;
  size_t i = (size_t)y * W + x; uint16_t v = d16[i];
  d[i] = v * 1e-3f; valid[i] = (v > 0) && (!alpha || alpha[i] >= alpha_min);
}
__device__ __forceinline__ int reflect101(int i, int n) { if (i < 0) i = -i; if (i >= n) i = 2 * n - 2 - i; return i; }
__global__ void depth_finish_kernel(int W, int H, const float* __restrict__ d, const uint8_t* __restrict__ valid, int erode, float edge_tol, float* __restrict__ out) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y; if (x >= W || y >= H) return;
  size_t i = (size_t)y * W + x;
  bool ok = valid[i];
  for (int dy = -erode; ok && dy <= erode; dy++) for (int dx = -erode; ok && dx <= erode; dx++) {
    int xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;  // outside the frame does not erode
    ok = valid[(size_t)yy * W + xx];
  }
  if (ok && edge_tol > 0.f) {
    float m[3][3];
    for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) m[dy + 1][dx + 1] = d[(size_t)reflect101(y + dy, H) * W + reflect101(x + dx, W)];
    float gx = (m[0][2] + 2.f * m[1][2] + m[2][2]) - (m[0][0] + 2.f * m[1][0] + m[2][0]);
    float gy = (m[2][0] + 2.f * m[2][1] + m[2][2]) - (m[0][0] + 2.f * m[0][1] + m[0][2]);
    ok = fmaxf(fabsf(gx), fabsf(gy)) * 0.125f < edge_tol;
  }
  out[i] = ok ? d[i] : 0.f;
}
__global__ void mask_prep_kernel(int W, int H, const uint16_t* __restrict__ d16, const uint8_t* __restrict__ alpha, int alpha_min, uint8_t* __restrict__ m) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y; if (x >= W || y >= H) return;
  size_t i = (size_t)y * W + x; m[i] = d16 ? (d16[i] > 0) : (alpha[i] >= alpha_min);
}
__global__ void dilate_kernel(int W, int H, const uint8_t* __restrict__ in, int r, uint8_t* __restrict__ out) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y; if (x >= W || y >= H) return;
  uint8_t v = 0;
  for (int dy = -r; !v && dy <= r; dy++) for (int dx = -r; !v && dx <= r; dx++) { int xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue; v = in[(size_t)yy * W + xx]; }
  out[(size_t)y * W + x] = v;
}

// ---- signed distance of a mesh on the grid ----
__global__ void near_flag_kernel(int3 cd, const int* __restrict__ cell_start, uint8_t* __restrict__ occ) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; size_t n = (size_t)cd.x * cd.y * cd.z; if (i >= n) return;
  occ[i] = cell_start[i + 1] > cell_start[i];
}
// Separable max over a (2r+1) window along one axis of a uint8 grid.
__global__ void max1d_kernel(int3 dims, const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int axis, int r) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; size_t n = (size_t)dims.x * dims.y * dims.z; if (i >= n) return;
  int z = (int)(i % dims.z); size_t q = i / dims.z; int y = (int)(q % dims.y), x = (int)(q / dims.y);
  int c[3] = {x, y, z}, d[3] = {dims.x, dims.y, dims.z};
  uint8_t v = 0;
  for (int k = -r; !v && k <= r; k++) { int cc = c[axis] + k; if (cc < 0 || cc >= d[axis]) continue; int xx = axis == 0 ? cc : x, yy = axis == 1 ? cc : y, zz = axis == 2 ? cc : z; v = in[((size_t)xx * dims.y + yy) * dims.z + zz]; }
  out[i] = v;
}
__global__ void sdf_query_kernel(VolumeGrid g, TriGridDev tg, const uint8_t* __restrict__ near, float band, int max_rings, float* __restrict__ out) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= g.n()) return;
  int x, y, z; float3 p = grid_point(g, i, x, y, z);
  int cx = (int)floorf((p.x - tg.lo.x) / tg.cell), cy = (int)floorf((p.y - tg.lo.y) / tg.cell), cz = (int)floorf((p.z - tg.lo.z) / tg.cell);
  bool maybe = cx >= 0 && cy >= 0 && cz >= 0 && cx < tg.dims.x && cy < tg.dims.y && cz < tg.dims.z && near[((size_t)cx * tg.dims.y + cy) * tg.dims.z + cz];
  float v = NAN;
  if (maybe) {
    float best = INFINITY; int best_t = -1; float3 best_p = p; int best_r = 0;
    for (int r = 0; r <= max_rings; r++) {
      if (r > 0 && best <= (float)(r - 1) * tg.cell) break;
      int x0 = cx - r, x1 = cx + r, y0 = cy - r, y1 = cy + r, z0 = cz - r, z1 = cz + r;
      for (int xx = max(x0, 0); xx <= min(x1, tg.dims.x - 1); xx++) for (int yy = max(y0, 0); yy <= min(y1, tg.dims.y - 1); yy++) for (int zz = max(z0, 0); zz <= min(z1, tg.dims.z - 1); zz++) {
        if (!(xx == x0 || xx == x1 || yy == y0 || yy == y1 || zz == z0 || zz == z1)) continue;
        int c = (xx * tg.dims.y + yy) * tg.dims.z + zz;
        for (int k = tg.cell_start[c]; k < tg.cell_start[c + 1]; k++) {
          int t = tg.items[k]; uint3 fc = tg.f[t];
          float3 bq; int reg; float3 q = closest_on_tri(p, tg.v[fc.x], tg.v[fc.y], tg.v[fc.z], bq, reg);
          float dd = len3(q - p);
          if (dd < best || (dd == best && t < best_t)) { best = dd; best_t = t; best_p = q; best_r = reg; }
        }
      }
    }
    if (best_t >= 0 && best <= band) v = best;
  }
  out[i] = v;
}
// Sign by the winding count along each z row: every triangle crossing the row's ray adds +1 when the ray enters
// (normal against +z) and -1 when it leaves; inside = count > 0. Overlapping closed shells (an arm pushed into the
// torso) stay inside, where a pseudo-normal at the closest feature would call the overlap outside. The exact
// distances in the band keep their magnitude; everything else takes +-SDF_FAR.
__global__ void winding_sign_kernel(VolumeGrid g, const float3* __restrict__ v, const uint3* __restrict__ f, const int* __restrict__ row_start, const int* __restrict__ row_items, float* __restrict__ sdf) {
  size_t row = blockIdx.x * (size_t)blockDim.x + threadIdx.x; size_t nrows = (size_t)g.dims.x * g.dims.y; if (row >= nrows) return;
  int x = (int)(row / g.dims.y), y = (int)(row % g.dims.y);
  // a hair off the grid line so a ray through a shared edge or vertex is not counted twice
  float px = g.lo.x + x * g.voxel + 1.7e-5f, py = g.lo.y + y * g.voxel + 2.3e-5f;
  constexpr int MAXC = 96; float cz[MAXC]; signed char cd[MAXC]; int nc = 0;
  for (int k = row_start[row]; k < row_start[row + 1]; k++) {
    uint3 fc = f[row_items[k]]; float3 a = v[fc.x], b = v[fc.y], c = v[fc.z];
    float area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    if (fabsf(area) < 1e-14f) continue;
    float inv = 1.f / area;
    float w0 = ((b.x - px) * (c.y - py) - (c.x - px) * (b.y - py)) * inv, w1 = ((c.x - px) * (a.y - py) - (a.x - px) * (c.y - py)) * inv, w2 = 1.f - w0 - w1;
    if (w0 < 0.f || w1 < 0.f || w2 < 0.f) continue;
    float z = w0 * a.z + w1 * b.z + w2 * c.z;
    signed char dir = area > 0.f ? -1 : 1;  // area > 0: the normal's z > 0, the +z ray leaves the solid
    if (nc < MAXC) { int j = nc++; while (j > 0 && cz[j - 1] > z) { cz[j] = cz[j - 1]; cd[j] = cd[j - 1]; j--; } cz[j] = z; cd[j] = dir; }
  }
  float* p = sdf + row * g.dims.z; int count = 0, next = 0;
  for (int k = 0; k < g.dims.z; k++) {
    float z = g.lo.z + k * g.voxel;
    while (next < nc && cz[next] <= z) count += cd[next++];
    float sign = count > 0 ? -1.f : 1.f; float val = p[k];
    p[k] = isnan(val) ? sign * SDF_FAR : sign * fabsf(val);
  }
}

// ---- cavity fill: line sweeps of an "outside" flag along +-x, +-y, +-z until nothing changes ----
__global__ void outside_init_kernel(VolumeGrid g, const float* __restrict__ vol, uint8_t* __restrict__ out) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= g.n()) return;
  int x, y, z; grid_point(g, i, x, y, z);
  bool border = x == 0 || y == 0 || z == 0 || x == g.dims.x - 1 || y == g.dims.y - 1 || z == g.dims.z - 1;
  out[i] = border && vol[i] >= 0.f;
}
// One thread per line along `axis`, sweeping both directions.
__global__ void outside_sweep_kernel(VolumeGrid g, const float* __restrict__ vol, uint8_t* __restrict__ out, int axis, int* __restrict__ changed) {
  size_t line = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  int d[3] = {g.dims.x, g.dims.y, g.dims.z}; int a1 = (axis + 1) % 3, a2 = (axis + 2) % 3;
  size_t nlines = (size_t)d[a1] * d[a2]; if (line >= nlines) return;
  int c[3]; c[a1] = (int)(line / d[a2]); c[a2] = (int)(line % d[a2]);
  size_t stride = axis == 0 ? (size_t)g.dims.y * g.dims.z : axis == 1 ? (size_t)g.dims.z : 1;
  c[axis] = 0; size_t base = ((size_t)c[0] * g.dims.y + c[1]) * g.dims.z + c[2];
  int n = d[axis]; int ch = 0;
  bool prev = out[base];
  for (int k = 1; k < n; k++) { size_t i = base + k * stride; bool o = out[i]; if (!o && prev && vol[i] >= 0.f) { out[i] = 1; o = true; ch++; } prev = o; }
  for (int k = n - 2; k >= 0; k--) { size_t i = base + k * stride; bool o = out[i]; if (!o && prev && vol[i] >= 0.f) { out[i] = 1; o = true; ch++; } prev = o; }
  if (ch) atomicAdd(changed, ch);
}
__global__ void cavity_fill_kernel(VolumeGrid g, float* __restrict__ vol, const uint8_t* __restrict__ out, int* __restrict__ filled) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= g.n()) return;
  if (!out[i] && vol[i] >= 0.f) { vol[i] = -1.f; atomicAdd(filled, 1); }
}

// ---- marching cubes ----
__device__ __forceinline__ bool inside(float v) { return v < 0.f; }
__device__ __forceinline__ size_t pidx(const VolumeGrid& g, int x, int y, int z) { return ((size_t)x * g.dims.y + y) * g.dims.z + z; }

// Per grid point: bit k set when the edge along axis k (x, y, z) from this point changes sign.
__global__ void point_flags_kernel(VolumeGrid g, const float* __restrict__ vol, uint8_t* __restrict__ flags) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= g.n()) return;
  int x, y, z; grid_point(g, i, x, y, z);
  bool a = inside(vol[i]); uint8_t f = 0;
  if (x + 1 < g.dims.x && inside(vol[pidx(g, x + 1, y, z)]) != a) f |= 1;
  if (y + 1 < g.dims.y && inside(vol[pidx(g, x, y + 1, z)]) != a) f |= 2;
  if (z + 1 < g.dims.z && inside(vol[pidx(g, x, y, z + 1)]) != a) f |= 4;
  flags[i] = f;
}
struct FlagNonZero { const uint8_t* f; __device__ bool operator()(int i) const { return f[i] != 0; } };
__device__ __forceinline__ int cube_index(const VolumeGrid& g, const float* vol, int x, int y, int z) {
  int ci = 0;
  if (inside(vol[pidx(g, x, y, z)])) ci |= 1; if (inside(vol[pidx(g, x + 1, y, z)])) ci |= 2;
  if (inside(vol[pidx(g, x + 1, y + 1, z)])) ci |= 4; if (inside(vol[pidx(g, x, y + 1, z)])) ci |= 8;
  if (inside(vol[pidx(g, x, y, z + 1)])) ci |= 16; if (inside(vol[pidx(g, x + 1, y, z + 1)])) ci |= 32;
  if (inside(vol[pidx(g, x + 1, y + 1, z + 1)])) ci |= 64; if (inside(vol[pidx(g, x, y + 1, z + 1)])) ci |= 128;
  return ci;
}
struct CellActive {
  VolumeGrid g; const float* vol;
  __device__ bool operator()(int c) const {
    int3 cd = make_int3(g.dims.x - 1, g.dims.y - 1, g.dims.z - 1);
    int z = c % cd.z; int q = c / cd.z; int y = q % cd.y, x = q / cd.y;
    int ci = cube_index(g, vol, x, y, z); return ci != 0 && ci != 255;
  }
};
__global__ void point_counts_kernel(int n, const int* __restrict__ pts, const uint8_t* __restrict__ flags, int* __restrict__ counts) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; counts[i] = __popc(flags[pts[i]]);
}
__global__ void emit_vertices_kernel(VolumeGrid g, const float* __restrict__ vol, int n, const int* __restrict__ pts, const uint8_t* __restrict__ flags, const int* __restrict__ base, float3* __restrict__ out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  size_t pi = pts[i]; int x, y, z; float3 p0 = grid_point(g, pi, x, y, z); float v0 = vol[pi]; uint8_t f = flags[pi]; int o = base[i];
  for (int a = 0; a < 3; a++) {
    if (!(f & (1 << a))) continue;
    size_t pj = a == 0 ? pidx(g, x + 1, y, z) : a == 1 ? pidx(g, x, y + 1, z) : pidx(g, x, y, z + 1);
    float v1 = vol[pj]; float t = v0 / (v0 - v1); t = fminf(fmaxf(t, 0.f), 1.f);
    float3 p = p0; if (a == 0) p.x += t * g.voxel; else if (a == 1) p.y += t * g.voxel; else p.z += t * g.voxel;
    out[o++] = p;
  }
}
__constant__ int8_t c_tri_table[256][16];
__constant__ uint8_t c_num_tris[256];
__global__ void cell_counts_kernel(VolumeGrid g, const float* __restrict__ vol, int n, const int* __restrict__ cells, int* __restrict__ counts) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  int3 cd = make_int3(g.dims.x - 1, g.dims.y - 1, g.dims.z - 1); int c = cells[i];
  int z = c % cd.z; int q = c / cd.z; int y = q % cd.y, x = q / cd.y;
  counts[i] = c_num_tris[cube_index(g, vol, x, y, z)];
}
__device__ __forceinline__ int find_point(const int* pts, int n, int key) {  // lower_bound; pts is sorted and contains key
  int lo = 0, hi = n; while (lo < hi) { int mid = (lo + hi) >> 1; if (pts[mid] < key) lo = mid + 1; else hi = mid; } return lo;
}
__global__ void emit_triangles_kernel(VolumeGrid g, const float* __restrict__ vol, int ncells, const int* __restrict__ cells, const int* __restrict__ tbase,
                                      int npts, const int* __restrict__ pts, const uint8_t* __restrict__ flags, const int* __restrict__ vbase, uint3* __restrict__ out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= ncells) return;
  int3 cd = make_int3(g.dims.x - 1, g.dims.y - 1, g.dims.z - 1); int c = cells[i];
  int z = c % cd.z; int q = c / cd.z; int y = q % cd.y, x = q / cd.y;
  int ci = cube_index(g, vol, x, y, z); int nt = c_num_tris[ci]; int o = tbase[i];
  // edge -> (point offset dx, dy, dz, axis)
  const int8_t E[12][4] = {{0,0,0,0},{1,0,0,1},{0,1,0,0},{0,0,0,1},{0,0,1,0},{1,0,1,1},{0,1,1,0},{0,0,1,1},{0,0,0,2},{1,0,0,2},{1,1,0,2},{0,1,0,2}};
  for (int t = 0; t < nt; t++) {
    unsigned vid[3];
    for (int k = 0; k < 3; k++) {
      int e = c_tri_table[ci][t * 3 + k];
      size_t pi = pidx(g, x + E[e][0], y + E[e][1], z + E[e][2]); int axis = E[e][3];
      int pos = find_point(pts, npts, (int)pi); uint8_t f = flags[pi];
      int before = __popc(f & ((1 << axis) - 1));
      vid[k] = (unsigned)(vbase[pos] + before);
    }
    out[o + t] = make_uint3(vid[0], vid[2], vid[1]);  // the table winds for negative-outside; ours is negative-inside
  }
}
// Orientation vote: +1 when the face normal points towards increasing volume values, sampled 1.5 voxels either side.
__global__ void orient_vote_kernel(VolumeGrid g, const float* __restrict__ vol, int nf, const float3* __restrict__ v, const uint3* __restrict__ f, int* __restrict__ votes) {
  int t = blockIdx.x * blockDim.x + threadIdx.x; if (t >= nf) return;
  uint3 fc = f[t]; float3 a = v[fc.x], b = v[fc.y], c = v[fc.z];
  float3 e1 = b - a, e2 = c - a; float3 n = make_float3(e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x);
  float l = len3(n); if (l < 1e-30f) { votes[t] = 0; return; } n = n * (1.f / l);
  float3 cen = (a + b + c) * (1.f / 3.f); float step = 1.5f * g.voxel;
  auto sample = [&](float3 p) { int x = min(max((int)lrintf((p.x - g.lo.x) / g.voxel), 0), g.dims.x - 1), y = min(max((int)lrintf((p.y - g.lo.y) / g.voxel), 0), g.dims.y - 1), z = min(max((int)lrintf((p.z - g.lo.z) / g.voxel), 0), g.dims.z - 1); return vol[pidx(g, x, y, z)]; };
  float fwd = sample(cen + n * step), bwd = sample(cen - n * step);
  votes[t] = fwd > bwd ? 1 : (fwd < bwd ? -1 : 0);
}

template <typename T> void cub_scan(DevBuf<unsigned char>& tmp, const T* in, T* out, int n, cudaStream_t stream) {
  size_t bytes = 0; cub::DeviceScan::ExclusiveSum(nullptr, bytes, in, out, n, stream); tmp.reserve(bytes);
  cub::DeviceScan::ExclusiveSum(tmp.ptr, bytes, in, out, n, stream); CUDA_KERNEL_CHECK();
}
template <typename Op> int cub_select(DevBuf<unsigned char>& tmp, DevBuf<int>& out, int n, Op op, cudaStream_t stream) {
  DevBuf<int> d_num; d_num.reserve(1); thrust::counting_iterator<int> it(0);
  size_t bytes = 0; cub::DeviceSelect::If(nullptr, bytes, it, out.ptr, d_num.ptr, n, op, stream); tmp.reserve(bytes);
  cub::DeviceSelect::If(tmp.ptr, bytes, it, out.ptr, d_num.ptr, n, op, stream); CUDA_KERNEL_CHECK();
  return d_num.download(1, stream)[0];
}

}  // namespace

void tsdf_integrate(const VolumeGrid& g, float* sdf, float* wgt, const uint8_t* protect, const std::vector<FuseView>& views, float trunc, cudaStream_t stream) {
  for (size_t off = 0; off < views.size(); off += MAX_BATCH) {
    int nb = (int)std::min<size_t>(MAX_BATCH, views.size() - off);
    ViewDev h[MAX_BATCH];
    for (int k = 0; k < nb; k++) { h[k].cam = views[off + k].cam; h[k].depth = views[off + k].depth; h[k].weight = views[off + k].weight; h[k].allowed = views[off + k].allowed_in_protect; }
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_views, h, sizeof(ViewDev) * nb, 0, cudaMemcpyHostToDevice, stream));
    integrate_kernel<<<div_up(g.n(), 256), 256, 0, stream>>>(g, sdf, wgt, protect, nb, trunc); CUDA_KERNEL_CHECK();
  }
}
void tsdf_carve_count(const VolumeGrid& g, uint8_t* carved, const std::vector<CarveView>& views, cudaStream_t stream) {
  for (size_t off = 0; off < views.size(); off += MAX_BATCH) {
    int nb = (int)std::min<size_t>(MAX_BATCH, views.size() - off);
    CarveDev h[MAX_BATCH];
    for (int k = 0; k < nb; k++) { h[k].cam = views[off + k].cam; h[k].mask = views[off + k].mask; }
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_carve, h, sizeof(CarveDev) * nb, 0, cudaMemcpyHostToDevice, stream));
    carve_kernel<<<div_up(g.n(), 256), 256, 0, stream>>>(g, carved, nb); CUDA_KERNEL_CHECK();
  }
}

void mesh_signed_distance(const VolumeGrid& g, const TriMesh& m, float band, float* out, cudaStream_t stream) {
  TriGrid tg; float cell = std::max(band / 2.f, 2.f * g.voxel);
  tg.build(m, cell, false, stream);
  int rings = (int)std::ceil(band / cell);   // after rings 0..R every unexamined triangle is >= R cells away, i.e. beyond the band
  // Cells with any triangle within `rings` cells, so the far voxels skip the search.
  size_t nc = (size_t)tg.dims.x * tg.dims.y * tg.dims.z;
  DevBuf<uint8_t> occ, tmp; occ.reserve(nc); tmp.reserve(nc);
  near_flag_kernel<<<div_up(nc, 256), 256, 0, stream>>>(tg.dims, tg.cell_start, occ); CUDA_KERNEL_CHECK();
  for (int axis = 0; axis < 3; axis++) { max1d_kernel<<<div_up(nc, 256), 256, 0, stream>>>(tg.dims, occ, tmp, axis, rings); CUDA_KERNEL_CHECK(); std::swap(occ.ptr, tmp.ptr); }
  sdf_query_kernel<<<div_up(g.n(), 128), 128, 0, stream>>>(g, tg.dev(), occ, band, rings, out); CUDA_KERNEL_CHECK();
  // Triangles per z row (the rows are the grid lines (x, y)), for the winding count.
  size_t nrows = (size_t)g.dims.x * g.dims.y;
  std::vector<std::pair<int, int>> pairs; pairs.reserve(m.nf() * 8);
  for (size_t t = 0; t < m.nf(); t++) {
    float x0 = INFINITY, x1 = -INFINITY, y0 = INFINITY, y1 = -INFINITY;
    for (int k = 0; k < 3; k++) { const float* p = &m.v[m.f[t * 3 + k] * 3]; x0 = std::min(x0, p[0]); x1 = std::max(x1, p[0]); y0 = std::min(y0, p[1]); y1 = std::max(y1, p[1]); }
    int ix0 = std::max((int)std::ceil((x0 - g.lo.x) / g.voxel) - 1, 0), ix1 = std::min((int)std::floor((x1 - g.lo.x) / g.voxel) + 1, g.dims.x - 1);
    int iy0 = std::max((int)std::ceil((y0 - g.lo.y) / g.voxel) - 1, 0), iy1 = std::min((int)std::floor((y1 - g.lo.y) / g.voxel) + 1, g.dims.y - 1);
    for (int x = ix0; x <= ix1; x++) for (int y = iy0; y <= iy1; y++) pairs.emplace_back(x * g.dims.y + y, (int)t);
  }
  std::sort(pairs.begin(), pairs.end());
  std::vector<int> rs(nrows + 1, 0), ri(pairs.size());
  for (size_t i = 0; i < pairs.size(); i++) { rs[pairs[i].first + 1]++; ri[i] = pairs[i].second; }
  for (size_t r = 0; r < nrows; r++) rs[r + 1] += rs[r];
  DevBuf<int> d_rs, d_ri; d_rs.upload(rs, stream); d_ri.upload(ri, stream);
  winding_sign_kernel<<<div_up(nrows, 128), 128, 0, stream>>>(g, tg.v, tg.f, d_rs, d_ri, out); CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

size_t fill_cavities(const VolumeGrid& g, float* vol, cudaStream_t stream) {
  DevBuf<uint8_t> out; out.reserve(g.n()); DevBuf<int> changed; changed.reserve(1);
  outside_init_kernel<<<div_up(g.n(), 256), 256, 0, stream>>>(g, vol, out); CUDA_KERNEL_CHECK();
  int d[3] = {g.dims.x, g.dims.y, g.dims.z};
  for (int sweep = 0; sweep < 1000; sweep++) {
    changed.zero(stream);
    for (int axis = 0; axis < 3; axis++) {
      size_t nlines = (size_t)d[(axis + 1) % 3] * d[(axis + 2) % 3];
      outside_sweep_kernel<<<div_up(nlines, 128), 128, 0, stream>>>(g, vol, out, axis, changed); CUDA_KERNEL_CHECK();
    }
    if (changed.download(1, stream)[0] == 0) break;
  }
  changed.zero(stream);
  cavity_fill_kernel<<<div_up(g.n(), 256), 256, 0, stream>>>(g, vol, out, changed); CUDA_KERNEL_CHECK();
  return (size_t)changed.download(1, stream)[0];
}

TriMesh marching_cubes(const VolumeGrid& g, const float* vol, cudaStream_t stream) {
  static bool tables_loaded = false;
  if (!tables_loaded) {
    uint8_t nt[256]; for (int c = 0; c < 256; c++) { int k = 0; while (k < 16 && MC_TRI_TABLE[c][k] >= 0) k++; nt[c] = (uint8_t)(k / 3); }
    CUDA_CHECK(cudaMemcpyToSymbol(c_tri_table, MC_TRI_TABLE, sizeof(MC_TRI_TABLE)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_num_tris, nt, sizeof(nt)));
    tables_loaded = true;
  }
  if (g.n() > (size_t)INT32_MAX) fail("marching cubes: grid too large (%zu points)", g.n());
  int npts_all = (int)g.n(); int3 cd = make_int3(g.dims.x - 1, g.dims.y - 1, g.dims.z - 1); int ncells_all = cd.x * cd.y * cd.z;
  DevBuf<uint8_t> flags; flags.reserve(g.n());
  point_flags_kernel<<<div_up(g.n(), 256), 256, 0, stream>>>(g, vol, flags); CUDA_KERNEL_CHECK();
  DevBuf<unsigned char> tmp; DevBuf<int> pts, cells; pts.reserve(g.n() / 8 + 1024); cells.reserve(g.n() / 8 + 1024);
  // Bounded scratch: the selections write at most as many entries as there are active points/cells; grow when short.
  int npts = 0, ncells = 0;
  for (;;) { npts = cub_select(tmp, pts, npts_all, FlagNonZero{flags}, stream); if ((size_t)npts <= pts.count) break; pts.reserve((size_t)npts + 1024); }
  for (;;) { ncells = cub_select(tmp, cells, ncells_all, CellActive{g, vol}, stream); if ((size_t)ncells <= cells.count) break; cells.reserve((size_t)ncells + 1024); }
  TriMesh m;
  if (npts == 0 || ncells == 0) return m;
  DevBuf<int> pcount, vbase, tcount, tbase; pcount.reserve(npts + 1); vbase.reserve(npts + 1); tcount.reserve(ncells + 1); tbase.reserve(ncells + 1);
  point_counts_kernel<<<div_up(npts, 256), 256, 0, stream>>>(npts, pts, flags, pcount); CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaMemsetAsync(pcount.ptr + npts, 0, sizeof(int), stream));
  cub_scan(tmp, pcount.ptr, vbase.ptr, npts + 1, stream);
  cell_counts_kernel<<<div_up(ncells, 256), 256, 0, stream>>>(g, vol, ncells, cells, tcount); CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaMemsetAsync(tcount.ptr + ncells, 0, sizeof(int), stream));
  cub_scan(tmp, tcount.ptr, tbase.ptr, ncells + 1, stream);
  int nv = vbase.download(npts + 1, stream)[npts], nf = tbase.download(ncells + 1, stream)[ncells];
  DevBuf<float3> verts; DevBuf<uint3> faces; verts.reserve(nv); faces.reserve(nf);
  emit_vertices_kernel<<<div_up(npts, 256), 256, 0, stream>>>(g, vol, npts, pts, flags, vbase, verts); CUDA_KERNEL_CHECK();
  emit_triangles_kernel<<<div_up(ncells, 128), 128, 0, stream>>>(g, vol, ncells, cells, tbase, npts, pts, flags, vbase, faces); CUDA_KERNEL_CHECK();
  // Orientation: the table's winding is fixed, so one global vote decides the flip.
  DevBuf<int> votes; votes.reserve(nf);
  orient_vote_kernel<<<div_up(nf, 256), 256, 0, stream>>>(g, vol, nf, verts, faces, votes); CUDA_KERNEL_CHECK();
  std::vector<int> hv = votes.download(nf, stream); long long sum = 0; for (int v : hv) sum += v;
  std::vector<float3> V = verts.download(nv, stream); std::vector<uint3> F = faces.download(nf, stream);
  m.v.resize((size_t)nv * 3); m.f.resize((size_t)nf * 3);
  for (int i = 0; i < nv; i++) { m.v[i * 3] = V[i].x; m.v[i * 3 + 1] = V[i].y; m.v[i * 3 + 2] = V[i].z; }
  bool flip = sum < 0;
  for (int i = 0; i < nf; i++) { m.f[i * 3] = F[i].x; m.f[i * 3 + 1] = flip ? F[i].z : F[i].y; m.f[i * 3 + 2] = flip ? F[i].y : F[i].z; }
  if (flip) log_info("marching cubes: flipped the winding (outward vote %.2f)", 0.5 + 0.5 * (double)sum / std::max(nf, 1));
  return m;
}

void keep_large_components(TriMesh& m, float min_frac) {
  size_t nv = m.nv(), nf = m.nf();
  std::vector<uint32_t> parent(nv); std::iota(parent.begin(), parent.end(), 0u);
  auto find = [&](uint32_t a) { while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
  auto unite = [&](uint32_t a, uint32_t b) { a = find(a); b = find(b); if (a != b) parent[std::max(a, b)] = std::min(a, b); };
  for (size_t t = 0; t < nf; t++) { unite(m.f[t * 3], m.f[t * 3 + 1]); unite(m.f[t * 3], m.f[t * 3 + 2]); }
  std::vector<double> area(nv, 0.0);
  for (size_t t = 0; t < nf; t++) {
    const float* a = &m.v[m.f[t * 3] * 3]; const float* b = &m.v[m.f[t * 3 + 1] * 3]; const float* c = &m.v[m.f[t * 3 + 2] * 3];
    double e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    double n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    area[find(m.f[t * 3])] += 0.5 * std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  }
  double amax = 0; size_t ncomp = 0; for (size_t i = 0; i < nv; i++) if (find((uint32_t)i) == i && area[i] > 0) { amax = std::max(amax, area[i]); ncomp++; }
  std::vector<uint8_t> keep_v(nv, 0); size_t kept = 0;
  for (size_t i = 0; i < nv; i++) { uint32_t r = find((uint32_t)i); keep_v[i] = area[r] >= min_frac * amax; }
  for (size_t i = 0; i < nv; i++) if (find((uint32_t)i) == i && area[i] > 0 && keep_v[i]) kept++;
  // Compact faces, then vertices (unreferenced ones go).
  std::vector<uint32_t> newf; newf.reserve(m.f.size());
  for (size_t t = 0; t < nf; t++) if (keep_v[m.f[t * 3]]) { newf.push_back(m.f[t * 3]); newf.push_back(m.f[t * 3 + 1]); newf.push_back(m.f[t * 3 + 2]); }
  std::vector<uint32_t> remap(nv, UINT32_MAX); std::vector<float> newv; std::vector<uint8_t> newc; std::vector<float> newn; uint32_t nn = 0;
  bool col = m.has_colour(), nrm = m.has_normals();
  for (uint32_t& i : newf) {
    if (remap[i] == UINT32_MAX) { remap[i] = nn++; newv.insert(newv.end(), &m.v[i * 3], &m.v[i * 3] + 3); if (col) newc.insert(newc.end(), &m.col[i * 3], &m.col[i * 3] + 3); if (nrm) newn.insert(newn.end(), &m.nrm[i * 3], &m.nrm[i * 3] + 3); }
    i = remap[i];
  }
  log_info("components kept %zu of %zu -> V %u F %zu", kept, ncomp, nn, newf.size() / 3);
  m.v = std::move(newv); m.f = std::move(newf); m.col = std::move(newc); m.nrm = std::move(newn);
}

// ---------------------------------------------------------------------------------------------------------------------
int mesh_fuse_main(int argc, char** argv) {
  std::string out, bbox_from, prior_path, prior_mode = "fill", depth_suffix = ".zfirst.png", protect_groups = "none", dump_prior, dump_sdf;
  std::vector<std::vector<std::string>> views, carves, weights, protect;
  float bbox[6] = {0, 0, 0, 0, 0, 0}; bool have_bbox = false; std::string bbox_s;
  float bbox_margin = 0.12f, voxel = 0.002f, trunc = 0.f, edge_tol = 0.01f, prior_offset = 0.f, min_comp = 0.05f, protect_band = 0.02f, protect_outside = 0.005f, min_weight = 1.f;
  int erode = 2, alpha_min = 128, carve_min = 3, carve_dilate = 2, device = 0; bool keep_cavities = false;
  ArgParser ap("TSDF fusion of the splat's depth probes with silhouette carving and a body-mesh prior\n\nUsage: b2ctrain mesh-fuse --output OUT.ply --views CAMS DIR [--views ...] [--carve CAMS DIR ...] [OPTIONS]");
  ap.s("output", "OUT.ply", "The fused mesh", out)
    .multi("views", "CAMS DIR", 2, "A camera list and a probe directory (<stem>.zfirst.png depth in mm + RGBA <name> whose alpha is the mask); repeatable, one group each", views)
    .multi("weight", "W", 1, "Fusion weight of the --views group in the same position (default 1)", weights)
    .multi("carve", "CAMS DIR", 2, "A camera list and a directory of masks for carving: a probe directory (the depth's coverage is the mask) or RGBA/grey images", carves)
    .s("bbox", "x0,y0,z0,x1,y1,z1", "The grid's extent", bbox_s)
    .s("bbox-from", "MESH", "The grid's extent from this mesh's bounds plus --bbox-margin", bbox_from)
    .f("bbox-margin", "M", "Margin around --bbox-from's bounds [default: 0.12]", bbox_margin)
    .f("voxel", "M", "Voxel size [default: 0.002]", voxel)
    .f("trunc", "M", "Truncation distance [default: 4 voxels]", trunc)
    .f("edge-tol", "M", "Reject depth pixels where the depth gradient (Sobel/8) exceeds this: silhouette rims and layer edges [default: 0.01]", edge_tol)
    .i("erode", "PX", "Erode every view's valid mask by this radius [default: 2]", erode)
    .i("alpha-min", "A", "Alpha threshold of the RGBA masks [default: 128]", alpha_min)
    .i("carve-min", "N", "Carve a voxel once this many views see it outside their mask [default: 3]", carve_min)
    .i("carve-dilate", "PX", "Dilate the carve masks by this radius [default: 2]", carve_dilate)
    .s("depth-suffix", "SUFFIX", "Depth file suffix in the probe directories [default: .zfirst.png]", depth_suffix)
    .s("prior", "MESH", "Body mesh (the SAM body) whose signed distance fills what the views never saw", prior_path)
    .s("prior-mode", "MODE", "fill: unobserved voxels take the prior; union: also the observed ones take min(fused, prior); protect: fill, with --protect required [default: fill]", prior_mode)
    .f("prior-offset", "M", "Shift the prior's surface: negative = inward [default: 0]", prior_offset)
    .f("min-weight", "W", "Voxels with less accumulated weight count as unobserved [default: 1]", min_weight)
    .multi("protect", "PLY RADIUS", 2, "Face protection: within RADIUS of the ply's points (and not more than --protect-outside outside the prior) the views listed in --protect-groups are the only ones fused; with none, the prior alone shapes the region", protect)
    .s("protect-groups", "LIST", "Comma list of --views group indices that may fuse inside the protected region, or none [default: none]", protect_groups)
    .f("protect-band", "M", "Beyond the protected region, blend the fused sdf toward the prior over this distance so the two surfaces meet without a ledge [default: 0.02]", protect_band)
    .f("protect-outside", "M", "How far outside the prior's surface the protection reaches (hair beyond that stays the views') [default: 0.005]", protect_outside)
    .f("min-comp", "FRAC", "Drop connected components smaller than this fraction of the largest one's area [default: 0.05]", min_comp)
    .b("keep-cavities", "Do not solidify enclosed free-space regions inside the surface (by default they are filled so no interior sheet reaches the mesh)", keep_cavities)
    .s("dump-sdf", "FILE", "Debug: write the final volume (before marching cubes) as raw float32", dump_sdf)
    .s("dump-prior", "FILE", "Debug: write the prior's signed distance grid as raw float32 (x-major, +-1000 beyond the exact band)", dump_prior)
    .i("device", "N", "CUDA device [default: 0]", device);
  if (!ap.parse(argc, argv)) return 0;
  if (out.empty()) fail("--output is required");
  if (views.empty()) fail("at least one --views group is required");
  if (!bbox_s.empty()) { if (sscanf(bbox_s.c_str(), "%f,%f,%f,%f,%f,%f", bbox, bbox + 1, bbox + 2, bbox + 3, bbox + 4, bbox + 5) != 6) fail("invalid --bbox '%s'", bbox_s.c_str()); have_bbox = true; }
  if (!have_bbox && bbox_from.empty()) fail("--bbox or --bbox-from is required");
  if (trunc <= 0.f) trunc = 4.f * voxel;
  if (prior_mode != "fill" && prior_mode != "union" && prior_mode != "protect") fail("--prior-mode must be fill, union or protect");
  if (!protect.empty() && prior_path.empty()) fail("--protect needs --prior (the region is the prior's, and the reach test uses its distance)");
  CUDA_CHECK(cudaSetDevice(device));
  cudaStream_t stream = 0;
  double t0 = now_seconds();
  if (!have_bbox) {
    TriMesh bm = read_mesh(bbox_from);
    float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (size_t i = 0; i < bm.nv(); i++) for (int k = 0; k < 3; k++) { lo[k] = std::min(lo[k], bm.v[i * 3 + k]); hi[k] = std::max(hi[k], bm.v[i * 3 + k]); }
    for (int k = 0; k < 3; k++) { bbox[k] = lo[k] - bbox_margin; bbox[k + 3] = hi[k] + bbox_margin; }
  }
  VolumeGrid g = VolumeGrid::from_bbox(bbox, bbox + 3, voxel);
  log_info("grid %d x %d x %d = %.1f M voxels (%.3f, %.3f, %.3f) .. (%.3f, %.3f, %.3f), voxel %.4f, trunc %.4f", g.dims.x, g.dims.y, g.dims.z, g.n() / 1e6, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], voxel, trunc);
  DevBuf<float> sdf, wgt; sdf.reserve(g.n()); wgt.reserve(g.n()); sdf.zero(stream); wgt.zero(stream);

  // The prior's signed distance, needed first when a protected region is requested.
  DevBuf<float> prior;
  if (!prior_path.empty()) {
    TriMesh pm = read_mesh(prior_path); prior.reserve(g.n());
    float band = std::max(4.f * trunc, std::abs(prior_offset) + 2.f * trunc + 0.02f);
    mesh_signed_distance(g, pm, band, prior, stream);
    log_info("prior %s: %zu triangles, exact within %.3f m (%.1fs)", prior_path.c_str(), pm.nf(), band, now_seconds() - t0);
    if (!dump_prior.empty()) { std::vector<float> hp = prior.download(g.n(), stream); write_f32(dump_prior, hp.data(), hp.size()); }
  }
  // Protected region: voxels within RADIUS (a cube, as the reference's repeated 3x3x3 max-pool) of the cap's points,
  // and not further than --protect-outside outside the prior.
  DevBuf<uint8_t> prot; std::vector<int> pgroups; size_t nprot = 0;
  if (!protect.empty()) {
    std::vector<float> pts = read_points(protect[0][0]); float radius = strtof(protect[0][1].c_str(), nullptr);
    std::vector<uint8_t> occ(g.n(), 0);
    for (size_t i = 0; i < pts.size() / 3; i++) {
      int c[3]; bool in = true;
      for (int k = 0; k < 3; k++) { c[k] = (int)std::lround((pts[i * 3 + k] - (k == 0 ? g.lo.x : k == 1 ? g.lo.y : g.lo.z)) / voxel); int d = k == 0 ? g.dims.x : k == 1 ? g.dims.y : g.dims.z; if (c[k] < 0 || c[k] >= d) in = false; }
      if (in) occ[((size_t)c[0] * g.dims.y + c[1]) * g.dims.z + c[2]] = 1;
    }
    DevBuf<uint8_t> a, b; a.upload(occ, stream); b.reserve(g.n());
    int r = (int)std::ceil(radius / voxel);
    for (int axis = 0; axis < 3; axis++) { max1d_kernel<<<div_up(g.n(), 256), 256, 0, stream>>>(g.dims, a, b, axis, r); CUDA_KERNEL_CHECK(); std::swap(a.ptr, b.ptr); }
    std::vector<uint8_t> hp = a.download(g.n(), stream);
    if (prior.ptr) { std::vector<float> ps = prior.download(g.n(), stream); for (size_t i = 0; i < g.n(); i++) if (!(ps[i] <= protect_outside)) hp[i] = 0; }
    for (uint8_t v : hp) nprot += v;
    prot.upload(hp, stream);
    if (protect_groups != "none" && !protect_groups.empty()) { std::string s = protect_groups; for (size_t p = 0; p < s.size();) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); int gi = atoi(s.substr(p, q - p).c_str()); if (gi < 0) gi += (int)views.size(); pgroups.push_back(gi); p = q + 1; } }
    log_info("protected region: %zu voxels (%.0f cm^3), fused there only by groups [%s]", nprot, nprot * voxel * voxel * voxel * 1e6, protect_groups.c_str());
  }

  // ---- fuse the depth views, group by group, in batches of MAX_BATCH decoded on the host in parallel ----
  int nviews_total = 0, ncarve_total = 0;
  DevBuf<uint16_t> d16; DevBuf<uint8_t> alpha, valid; DevBuf<float> dtmp;
  std::vector<DevBuf<float>> slots(MAX_BATCH); std::vector<DevBuf<uint8_t>> mslots(MAX_BATCH); DevBuf<uint8_t> mtmp;
  struct Item { std::string depth_path, rgba_path; CameraGPU cam; float weight; int group; };
  DevBuf<uint8_t> carved; if (!carves.empty()) { carved.reserve(g.n()); carved.zero(stream); }
  auto run_items = [&](const std::vector<Item>& items, bool carve) {
    for (size_t off = 0; off < items.size(); off += MAX_BATCH) {
      int nb = (int)std::min<size_t>(MAX_BATCH, items.size() - off);
      struct Decoded { int W = 0, H = 0; std::vector<uint16_t> d; Image8 rgba; bool ok = false; };
      std::vector<Decoded> dec(nb);
      parallel_for(nb, [&](size_t k) {
        const Item& it = items[off + k]; Decoded& d = dec[k];
        if (!it.depth_path.empty()) d.ok = try_load_png16(it.depth_path, d.W, d.H, d.d);
        if (!it.rgba_path.empty()) {
          Image8 im;
          if (try_load_image8(it.rgba_path, 4, im)) {
            if (d.ok && (im.W != d.W || im.H != d.H)) fail("'%s': size differs from its depth", it.rgba_path.c_str());
            if (!d.ok) { d.W = im.W; d.H = im.H; d.ok = carve; }  // a carve mask may come from the image alone
            d.rgba = std::move(im);
          }
        }
      });
      std::vector<FuseView> fv; std::vector<CarveView> cv;
      for (int k = 0; k < nb; k++) {
        const Item& it = items[off + k]; Decoded& d = dec[k];
        if (!d.ok) { log_warn("missing %s", (it.depth_path.empty() ? it.rgba_path : it.depth_path).c_str()); continue; }
        if (d.W != it.cam.W || d.H != it.cam.H) fail("'%s': %d x %d does not match the camera list's %d x %d", it.depth_path.c_str(), d.W, d.H, it.cam.W, it.cam.H);
        size_t npx = (size_t)d.W * d.H; dim3 blk(16, 16), grd((d.W + 15) / 16, (d.H + 15) / 16);
        bool have_alpha = d.rgba.ok();
        if (have_alpha) { std::vector<uint8_t> a(npx); for (size_t i = 0; i < npx; i++) a[i] = d.rgba.C == 4 ? d.rgba.px[i * 4 + 3] : d.rgba.px[i]; alpha.upload(a, stream); }
        if (!carve) {
          d16.upload(d.d, stream); valid.reserve(npx); dtmp.reserve(npx); slots[k].reserve(npx);
          depth_prep_kernel<<<grd, blk, 0, stream>>>(d.W, d.H, d16, have_alpha ? alpha.ptr : nullptr, alpha_min, dtmp, valid); CUDA_KERNEL_CHECK();
          depth_finish_kernel<<<grd, blk, 0, stream>>>(d.W, d.H, dtmp, valid, erode, edge_tol, slots[k]); CUDA_KERNEL_CHECK();
          bool allowed = false; for (int gi : pgroups) if (gi == it.group) allowed = true;
          fv.push_back({it.cam, slots[k].ptr, it.weight, allowed});
        } else {
          bool from_depth = !d.d.empty();
          if (from_depth) d16.upload(d.d, stream);
          else if (!have_alpha) { log_warn("carve: '%s' has neither depth nor alpha", it.rgba_path.c_str()); continue; }
          mtmp.reserve(npx); mslots[k].reserve(npx);
          mask_prep_kernel<<<grd, blk, 0, stream>>>(d.W, d.H, from_depth ? d16.ptr : nullptr, have_alpha ? alpha.ptr : nullptr, alpha_min, mtmp); CUDA_KERNEL_CHECK();
          dilate_kernel<<<grd, blk, 0, stream>>>(d.W, d.H, mtmp, carve_dilate, mslots[k]); CUDA_KERNEL_CHECK();
          cv.push_back({it.cam, mslots[k].ptr});
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
      if (!carve) { tsdf_integrate(g, sdf, wgt, prot.ptr, fv, trunc, stream); nviews_total += (int)fv.size(); }
      else { tsdf_carve_count(g, carved, cv, stream); ncarve_total += (int)cv.size(); }
    }
  };
  for (size_t gi = 0; gi < views.size(); gi++) {
    CamSet cs = load_cams(views[gi][0]); float w = gi < weights.size() ? strtof(weights[gi][0].c_str(), nullptr) : 1.f;
    std::vector<Item> items;
    for (size_t i = 0; i < cs.size(); i++) items.push_back({views[gi][1] + "/" + file_stem(cs.names[i]) + depth_suffix, views[gi][1] + "/" + cs.names[i], CameraGPU::from(cs.cams[i], cs.W, cs.H), w, (int)gi});
    run_items(items, false);
  }
  for (size_t gi = 0; gi < carves.size(); gi++) {
    CamSet cs = load_cams(carves[gi][0]);
    std::vector<Item> items;
    for (size_t i = 0; i < cs.size(); i++) {
      std::string dp = carves[gi][1] + "/" + file_stem(cs.names[i]) + depth_suffix;
      items.push_back({file_exists(dp) ? dp : "", carves[gi][1] + "/" + cs.names[i], CameraGPU::from(cs.cams[i], cs.W, cs.H), 1.f, (int)gi});
    }
    run_items(items, true);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  log_info("fused %d depth views + %d carve masks (%.1fs)", nviews_total, ncarve_total, now_seconds() - t0);

  // ---- combine on the host: carve, band, prior fill (one pass over the grid; the grids are 400 MB each) ----
  std::vector<float> hs = sdf.download(g.n(), stream), hw = wgt.download(g.n(), stream);
  if (carved.ptr) {
    std::vector<uint8_t> hc = carved.download(g.n(), stream); size_t n = 0;
    for (size_t i = 0; i < g.n(); i++) if (hc[i] >= carve_min) { hs[i] = 1.f; hw[i] = 100.f; n++; }
    log_info("carved voxels %.1f%%", 100.0 * n / g.n());
  }
  if (prior.ptr) {
    std::vector<float> hp = prior.download(g.n(), stream);
    for (auto& v : hp) v = std::min(std::max((v - prior_offset) / trunc, -1.f), 1.f);
    if (prot.ptr && protect_band > 0.f) {
      // L-inf distance (in voxels) from the protected region, up to K: band weight 1 - k / (K + 1).
      int K = std::max(1, (int)std::lround(protect_band / voxel));
      DevBuf<uint8_t> a, b; a.reserve(g.n()); b.reserve(g.n());
      CUDA_CHECK(cudaMemcpyAsync(a.ptr, prot.ptr, g.n(), cudaMemcpyDeviceToDevice, stream));
      std::vector<uint8_t> dist(g.n(), 0); std::vector<uint8_t> prev = prot.download(g.n(), stream);
      for (int k = 1; k <= K; k++) {
        for (int axis = 0; axis < 3; axis++) { max1d_kernel<<<div_up(g.n(), 256), 256, 0, stream>>>(g.dims, a, b, axis, 1); CUDA_KERNEL_CHECK(); std::swap(a.ptr, b.ptr); }
        std::vector<uint8_t> cur = a.download(g.n(), stream);
        for (size_t i = 0; i < g.n(); i++) if (cur[i] && !prev[i]) dist[i] = (uint8_t)k;
        prev.swap(cur);
      }
      size_t nb = 0;
      for (size_t i = 0; i < g.n(); i++) if (dist[i]) { float bw = 1.f - (float)dist[i] / (K + 1); hs[i] = bw * hp[i] + (1.f - bw) * hs[i]; if (hw[i] < min_weight) hw[i] = 1.f; nb++; }
      log_info("protect band: %zu voxels over %d voxels, blended toward the prior", nb, K);
    }
    size_t nfill = 0;
    for (size_t i = 0; i < g.n(); i++) {
      bool unobs = hw[i] < min_weight;
      if (prior_mode == "union" && !unobs) hs[i] = std::min(hs[i], hp[i]);
      if (unobs) { hs[i] = hp[i]; hw[i] = 1.f; nfill++; }
    }
    log_info("prior filled %.1f%% of voxels (%s)", 100.0 * nfill / g.n(), prior_mode.c_str());
    prior.free();
  }
  for (size_t i = 0; i < g.n(); i++) if (hw[i] < min_weight) hs[i] = 1.f;  // unobserved = outside
  wgt.free(); prot.free(); carved.free();
  sdf.upload(hs, stream);
  if (!keep_cavities) { size_t nfill = fill_cavities(g, sdf, stream); log_info("cavities: %zu enclosed voxels solidified", nfill); }
  if (!dump_sdf.empty()) { std::vector<float> hv = sdf.download(g.n(), stream); write_f32(dump_sdf, hv.data(), hv.size()); }
  TriMesh m = marching_cubes(g, sdf, stream);
  log_info("mesh V %zu F %zu (%.1fs)", m.nv(), m.nf(), now_seconds() - t0);
  if (min_comp > 0.f) keep_large_components(m, min_comp);
  write_ply_mesh(out, m);
  log_info("wrote %s (%.1fs)", out.c_str(), now_seconds() - t0);
  return 0;
}

}  // namespace b2c
