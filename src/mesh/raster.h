#pragma once
// GPU mesh rasteriser for the meshification chain: per pixel the nearest triangle's depth, its id and
// perspective-correct barycentrics (the torch `gpu_raster.py` of the reference tools), plus the mesh-side helpers
// every stage needs: vertex normals, an adjacency CSR, and uniform grids for closest-triangle / nearest-point queries.
#include "gpu/render.h"
#include "dataset/mesh.h"
#include <vector>

namespace b2c {

// CameraGPU with the intrinsics scaled for an ss x supersampled raster (pixel centres at (i + 0.5) / ss).
CameraGPU scaled_camera(const CameraGPU& c, int ss);
// Camera z and pixel coordinates (centre convention i + 0.5) of a world point; returns z (<= 0: behind).
__host__ __device__ inline float project_point(const CameraGPU& c, float x, float y, float z, float& u, float& v) {
  float xc = c.R[0] * x + c.R[1] * y + c.R[2] * z + c.t[0];
  float yc = c.R[3] * x + c.R[4] * y + c.R[5] * z + c.t[1];
  float zc = c.R[6] * x + c.R[7] * y + c.R[8] * z + c.t[2];
  float zs = fabsf(zc) < 1e-9f ? 1e-9f : zc;
  u = c.fx * xc / zs + c.cx; v = c.fy * yc / zs + c.cy;
  return zc;
}

struct MeshRaster {
  DevBuf<float3> v; DevBuf<uint3> f; int nv = 0, nf = 0;
  DevBuf<unsigned long long> keys;   // per pixel (depth bits << 32 | triangle id), atomicMin resolves the nearest
  DevBuf<float> z;                   // per pixel camera depth, 0 = miss
  DevBuf<int> tri;                   // per pixel triangle id, -1 = miss
  DevBuf<float2> bary;               // per pixel perspective-correct (b1, b2) of the winning triangle (b0 = 1 - b1 - b2)
  int W = 0, H = 0;                  // of the last raster (already multiplied by ss)
  void upload(const TriMesh& m, cudaStream_t stream = 0);
  void set_vertices(const float3* dev_v, cudaStream_t stream = 0);  // replace positions in place (same count)
  // Rasterise at (cam.W * ss) x (cam.H * ss). `with_bary` also fills tri/bary (z is always filled).
  void raster(const CameraGPU& cam, int ss, bool with_bary, cudaStream_t stream = 0);
};

// Area-weighted vertex normals (open3d's compute_vertex_normals): unit length, zero for unreferenced vertices.
void vertex_normals_gpu(const float3* v, int nv, const uint3* f, int nf, float3* out, cudaStream_t stream = 0);
// Per-face unit normals.
void face_normals_gpu(const float3* v, const uint3* f, int nf, float3* out, cudaStream_t stream = 0);

// Unique undirected vertex adjacency in CSR form (neighbours of i are nbr[start[i] .. start[i+1]), sorted).
struct Adjacency {
  std::vector<int> start, nbr;
  DevBuf<int> d_start, d_nbr;
  void build(const TriMesh& m);
  void upload(cudaStream_t stream = 0);
  int max_degree() const;
};

// Uniform grid over the triangles of a mesh for closest-point queries. Every triangle is listed in every cell its
// bounding box overlaps; a query walks rings of cells outward until the best distance found is guaranteed exact.
struct TriGridDev {
  float3 lo; float cell; int3 dims;
  const int* cell_start; const int* items;     // CSR: items[cell_start[c] .. cell_start[c+1])
  const float3* v; const uint3* f;
  const float3* fn;                            // face normals (unit)
  const float3* vn;                            // angle-weighted vertex normals (for the sign of a distance)
  const float3* en;                            // [nf][3] edge normals: edge k of face is (k, k+1)
  int nf;
};
struct TriGrid {
  float3 lo, hi; float cell = 0; int3 dims{0, 0, 0};
  DevBuf<int> cell_start, items;
  DevBuf<float3> v, fn, vn, en; DevBuf<uint3> f;
  int nv = 0, nf = 0;
  // `cell` = the cell size; pseudo-normals are built when `with_normals` (needed for signed distances).
  void build(const TriMesh& m, float cell, bool with_normals, cudaStream_t stream = 0);
  TriGridDev dev() const;
};
// Closest point on the mesh for every query: triangle id (-1 when nothing within max_rings cells), the point, the
// distance, barycentrics (b0, b1, b2) and, with pseudo-normals, the sign (+1 outside).
void closest_points_gpu(const TriGridDev& g, const float3* q, int n, int max_rings, int* tri, float3* point, float* dist, float3* bary, float* sign, cudaStream_t stream = 0);

// Uniform grid over points (the face cap's Gaussians) for nearest-distance and count-within-radius queries.
struct PointGridDev { float3 lo; float cell; int3 dims; const int* cell_start; const int* items; const float3* p; int n; };
struct PointGrid {
  float3 lo, hi; float cell = 0; int3 dims{0, 0, 0};
  DevBuf<int> cell_start, items; DevBuf<float3> p; int n = 0;
  void build(const std::vector<float>& xyz, float cell, cudaStream_t stream = 0);
  PointGridDev dev() const;
};
// Distance to the nearest point within `radius` (INFINITY beyond it) and the number of points within `radius`.
void point_query_gpu(const PointGridDev& g, const float3* q, int n, float radius, float* dist, int* count, cudaStream_t stream = 0);

}  // namespace b2c
