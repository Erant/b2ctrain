#pragma once
// `b2ctrain lod-merge`: a splat LOD level by moment-matched merging (b2cgltf_lod_proposal.md 4.2).
//
// A C++ port of the uniform decimation of PlayCanvas splat-transform 3.10.1 (MIT License, Copyright (c) 2011-2026
// PlayCanvas Ltd.; src/lib/decimate-uniform/ and src/lib/decimate/moment-match.ts), the path `splat-transform -d N`
// runs by default. Per generation: every splat's 16 nearest neighbours, an edge cost to each, the 4 cheapest kept;
// a greedy disjoint matching in cost order (1024 linear buckets), then a chain closure that attaches the splats left
// free to a neighbour's group (up to 3, then 4 members), so a generation removes half the splats; each group is
// moment-matched into one splat (area * alpha weights, law-of-total-variance covariance, mass-conserving opacity
// capped at 1, SH as the weighted mean). Generations repeat until the count is reached exactly.
//
// The edge cost is the uniform path's (decimate-uniform/edge-cost-cpu.ts): a one-sample Monte-Carlo KL term plus the
// L2 over every SH coefficient. The adaptive path's closed-form L2 field cost (decimate/edge-cost-cpu.ts) was tried
// under this selection on crew-brazil and lost 2 dB before any fine-tune (2026-10-30), so it is not ported.
//
// b2c's merge rules, inside the merge: neighbours are searched within a seg_label class only, so no group crosses a
// class; a merged splat's seg_conf is its group's minimum; with a cage, a group whose merged extent (3 sigma along the
// major axis) exceeds `max_extent_faces` times the mean edge length of the cage face nearest its mean is not formed:
// a capped pair leaves the splat's candidate list (its cheapest allowed merges stay), a capped join is skipped. When
// the cap leaves too few merges to reach the count, the merge stops with an error rather than miss it. Only the splat
// itself and its labels are carried: evidence and b2crig's per-splat state are dropped, never averaged.
#include "ply.h"
#include <cstddef>
#include <vector>

namespace b2c {

// The canonical cage's faces, for the distance cap (host copy of the cage file's layers; gpu/cage.h).
struct CageFaces {
  std::vector<float> verts;                 // [nv][3] canonical
  std::vector<int> faces;                   // [nf][3]
  std::vector<int> face_layer;              // [nf]
  std::vector<unsigned> layer_classes;      // per layer: bit c = Sapiens2 class c
  std::vector<int> layer_v_off, layer_v_count;
};

struct MergeParams {
  size_t count = 0;
  const CageFaces* cage = nullptr;   // the distance cap, when set
  float max_extent_faces = 3.f;
};

struct MergeStats {
  int generations = 0;
  size_t cap_blocked = 0;   // merges the cage cap ruled out (candidate pairs and chain joins), over every generation
  size_t cap_held = 0;      // splats of the last generation left unmerged with a merge the cap ruled out
  size_t classes = 0;       // distinct seg_label classes in the input
};

SplatCloud lod_merge(const SplatCloud& in, const MergeParams& p, MergeStats* stats = nullptr);

// Moment-match `n` splats of `c` into row `o` of `out` (splat-transform's mergeGroup); out's labels, when it has
// them, take the first member's class and the members' minimum seg_conf.
void lod_merge_group(const SplatCloud& c, const int* members, int n, SplatCloud& out, size_t o);

// The edge cost between splats i and j of `c` (splat-transform's computeEdgeCostView with its one MC sample).
double lod_edge_cost(const SplatCloud& c, size_t i, size_t j);

int lod_merge_main(int argc, char** argv);
int lod_distill_main(int argc, char** argv);

}  // namespace b2c
