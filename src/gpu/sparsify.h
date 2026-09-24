#pragma once
#include "model.h"
#include <vector>

namespace b2c {

// GaussianSpa (Zhang et al., CVPR 2025, github.com/noodle-lab/GaussianSpa): simplification as an L0-constrained
// optimisation on the splats' opacities, solved by alternating two steps.
//   optimizing:  the photometric loss plus 0.5 * rho * ||o - z + u||^2 (the penalty's gradient is added to the opacity
//                logit in optim_kernel, through OptimParams::spa_*);
//   sparsifying: z = the projection of o + u onto "at most `keep` non-zero entries" (top-k by o + u, or zeroed below
//                the importance cut), and the dual update u += o - z.
// o is the effective opacity (sigmoid times the 3D-filter floor compensation), the one the rasteriser draws with and
// the one the refine prune and the export see. The number kept is fixed when the phase starts (a ratio of the splats
// alive then), so splats that die in the phase and are compacted away do not make later projections cut deeper.
// At the stop the bottom of the ranking is removed for good (Model::compact).
struct Sparsifier {
  DevBuf<float> z, u, score, keys, keys_sorted;
  DevBuf<uint32_t> flags, count;
  DevBuf<unsigned char> cub_tmp;
  uint32_t keep = 0;             // splats allowed to stay non-zero
  float rho = 5e-4f;
  bool by_importance = false;    // rank by `score` (Mini-Splatting's importance: blending weight over all views) instead of o + u
  bool active = false;

  // score[i] = acc[i * stride + offset] (the evidence accumulator's w_all).
  void set_importance(const float* acc, int stride, int offset, const Model& m, cudaStream_t stream);
  // Start the phase: keep = round((1 - ratio) * n), u = 0, z = proj(o).
  void begin(const Model& m, float ratio, cudaStream_t stream);
  // One sparsifying step: z = proj(o + u), u += o - z.
  void update(const Model& m, cudaStream_t stream);
  // The final prune's keep flags [n]: the top `keep` splats by importance, or by o. Returns the number flagged.
  uint32_t final_flags(const Model& m, cudaStream_t stream);
  // Per-splat buffers to carry through Model::compact while the phase runs.
  std::vector<DevBuf<float>*> state() { return {&z, &u, &score}; }

 private:
  void project(const Model& m, bool update_u, cudaStream_t stream);
  // The largest key that does not make the top `keep` (-inf when every key does).
  float cut(int n, cudaStream_t stream);
};

}  // namespace b2c
