#pragma once
#include "gpu/render.h"
#include "train/init.h"
#include "ply.h"

namespace b2c {

struct RefineParams {
  uint32_t iter = 0, total = 30000;
  bool growth_allowed = true;
  uint32_t max_splats = 10000000;
  float growth_grad_threshold = 0.0025f, growth_select_fraction = 0.25f;
  float split_at_screen_size = 0.5f;
  float opac_decay = 0.004f;
  uint32_t seed = 42;
};
struct RefineStats { int pruned = 0, relocated = 0, grown = 0, split_oversized = 0; };

struct RefineState {
  Bounds bounds{};
  bool accumulate_min_scale = false;
  DevBuf<float> cam_pos, cam_focal; int n_cams = 0;
  // scratch
  DevBuf<uint32_t> keep, dead_flag, dead_idx, selected, above, tmp_idx, sel_parent, sel_child;
  DevBuf<float> keys, keys_sorted; DevBuf<uint32_t> vals, vals_sorted;
  DevBuf<unsigned char> cub_tmp;
  DevBuf<uint32_t> counts;
  PinnedBuf<uint32_t> h_counts;
  PinnedBuf<float> h_bounds;
  int refine_count = 0;

  void init(const Model& m, cudaStream_t stream);
  void set_cameras(const std::vector<float>& pos, const std::vector<float>& focal);
  // Recompute the per-splat 3D-filter floor f = sqrt(0.1) * min_v(dist / focal) into lscale.w.
  void update_min_scale(Model& m, cudaStream_t stream);
  // Fold the floor into the raw params (scales, opacity) and clear it.
  void bake_min_scale(Model& m, cudaStream_t stream);
  RefineStats run(Model& m, RenderCtx& ctx, const RefineParams& p, cudaStream_t stream);
  void update_bounds(const Model& m, cudaStream_t stream);
};

// Fold the model's current floor into a downloaded cloud (for export).
void bake_min_scale_cpu(SplatCloud& c, const Model& m, cudaStream_t stream);

}  // namespace b2c
