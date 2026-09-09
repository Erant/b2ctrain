#pragma once
#include "gpu/render.h"
#include "dataset/mesh.h"

namespace b2c {

// The body proxy mesh on the GPU and a per-view depth rasteriser for it.
// Without a mesh, a point cloud sampled on the body surface (b2crunner's points3D.txt is exactly that) stands in:
// every point becomes a surfel, a disc of `radius` in the tangent plane estimated from its neighbours, each pixel's
// ray is intersected with that plane, nearest depth wins, and the dilation below closes the small gaps. (A
// camera-facing disc would sit in front of the real surface wherever it is seen at a grazing angle.) Both can be
// loaded; the triangles and the surfels then share the depth buffer.
struct MeshGPU {
  DevBuf<float3> v; DevBuf<uint3> f;
  int nv = 0, nf = 0;
  DevBuf<float4> pts, nrm; int np = 0;  // (x, y, z, radius), (nx, ny, nz, 0)
  DevBuf<uint32_t> zbuf;   // scratch: float-as-uint depth for atomicMin
  DevBuf<float> depth;     // [W*H] front-face depth (camera z), +inf where the mesh is not hit
  void upload(const TriMesh& m, cudaStream_t stream = 0);
  // Normals are estimated here (PCA over the 12 nearest neighbours; O(n^2) on the CPU, fine for the ~10k points
  // b2crunner writes, slow beyond ~100k).
  void upload_points(const float* xyz, size_t n, float radius, cudaStream_t stream = 0);
  bool empty() const { return nf == 0 && np == 0; }
  // Rasterise the mesh's nearest depth for `cam` into `depth` at W x H, then dilate it by `dilate` pixels taking the
  // FARTHEST depth in the window (infinity wins), so a pixel's reference surface is the deepest one within `dilate`
  // pixels: near a silhouette or a fold the penalty only starts behind whichever surface is farther.
  void rasterize(const CameraGPU& cam, int W, int H, int dilate, cudaStream_t stream);
};

}  // namespace b2c
