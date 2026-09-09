#pragma once
#include "gpu/render.h"
#include "dataset/mesh.h"

namespace b2c {

// The body proxy mesh on the GPU and a per-view depth rasteriser for it.
struct MeshGPU {
  DevBuf<float3> v; DevBuf<uint3> f;
  int nv = 0, nf = 0;
  DevBuf<uint32_t> zbuf;   // scratch: float-as-uint depth for atomicMin
  DevBuf<float> depth;     // [W*H] front-face depth (camera z), +inf where the mesh is not hit
  void upload(const TriMesh& m, cudaStream_t stream = 0);
  // Rasterise the mesh's nearest depth for `cam` into `depth` at W x H, then dilate it by `dilate` pixels taking the
  // FARTHEST depth in the window (infinity wins), so a pixel's reference surface is the deepest one within `dilate`
  // pixels: near a silhouette or a fold the penalty only starts behind whichever surface is farther.
  void rasterize(const CameraGPU& cam, int W, int H, int dilate, cudaStream_t stream);
};

}  // namespace b2c
