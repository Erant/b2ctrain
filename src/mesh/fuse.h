#pragma once
// `b2ctrain mesh-fuse`: TSDF fusion of the splat's depth probes with silhouette carving and the SAM body as a signed
// distance prior (the reference is out/mesh/tools/tsdf_torch.py + body_sdf.py). The pieces are exposed for the tests.
#include "gpu/render.h"
#include "dataset/mesh.h"
#include <vector>

namespace b2c {

// A dense grid: point (i, j, k) sits at lo + (i, j, k) * voxel; index = (i * dims.y + j) * dims.z + k.
struct VolumeGrid {
  float3 lo{0, 0, 0}; int3 dims{0, 0, 0}; float voxel = 0.002f;
  __host__ __device__ size_t n() const { return (size_t)dims.x * dims.y * dims.z; }
  static VolumeGrid from_bbox(const float* lo, const float* hi, float voxel);
};

// One depth view for the integration: the camera, the depth map on the device (metres, 0 = no depth) and its weight.
struct FuseView { CameraGPU cam; const float* depth; float weight; bool allowed_in_protect; };
// One carve view: a uint8 mask on the device (1 = inside the (dilated) silhouette).
struct CarveView { CameraGPU cam; const uint8_t* mask; };

// Weighted running-mean TSDF update of `sdf`/`wgt` (both [n]) with every view; `protect` ([n] uint8, may be null):
// views not allowed there skip protected voxels. Any number of views: they are consumed in batches internally.
void tsdf_integrate(const VolumeGrid& g, float* sdf, float* wgt, const uint8_t* protect, const std::vector<FuseView>& views, float trunc, cudaStream_t stream = 0);
// Count, per voxel, the views that see it outside their mask (uint8, saturating).
void tsdf_carve_count(const VolumeGrid& g, uint8_t* carved, const std::vector<CarveView>& views, cudaStream_t stream = 0);

// Signed distance of a closed, consistently wound mesh on the grid: exact within `band` metres of the surface,
// +-FAR beyond it. The sign is the winding count along each z row (entering minus leaving crossings), so overlapping
// shells count as inside; the grid's z = 0 face must lie outside the mesh.
constexpr float SDF_FAR = 1000.f;
void mesh_signed_distance(const VolumeGrid& g, const TriMesh& m, float band, float* out, cudaStream_t stream = 0);

// Flood the "outside" (vol >= 0) from the grid's border faces and set every enclosed positive region to -1: interior
// bubbles (free space a see-through depth left inside the body) and the prior's surface inside the fused one would
// otherwise become interior sheets. Returns the number of voxels filled.
size_t fill_cavities(const VolumeGrid& g, float* vol, cudaStream_t stream = 0);

// Marching cubes of the zero level set of vol ([n], negative = inside): welded vertices, triangles wound so their
// normals point towards positive values.
TriMesh marching_cubes(const VolumeGrid& g, const float* vol, cudaStream_t stream = 0);

// Drops connected components whose area is below `min_frac` of the largest one's; unreferenced vertices go too.
void keep_large_components(TriMesh& m, float min_frac);

int mesh_fuse_main(int argc, char** argv);

}  // namespace b2c
