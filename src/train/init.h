#pragma once
#include "ply.h"
#include "dataset/views.h"
#include "cli.h"

namespace b2c {
// Build the initial splat cloud: init.ply if present, else points3D with kNN scales. Applies subsampling and the
// max_splats cap, and pads/truncates SH to cfg.sh_degree.
SplatCloud initial_splats(const Dataset& ds, const Config& cfg);
// Percentile bounding box of positions (brush's bounds_from_pos).
struct Bounds { float min[3], max[3]; float center[3]; float extent[3]; float median_size() const; float max_extent() const; };
Bounds bounds_from_pos(const float* pos, size_t n, float percentile);
// Median distance from a point to its nearest neighbour (estimated on up to `samples` query points).
float median_nn_distance(const float* pos, size_t n, size_t samples = 2048);
}
