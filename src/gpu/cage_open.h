#pragma once
#include "gpu/cage.h"
#include "gpu/render.h"
#include "ply.h"
#include <string>
#include <vector>

namespace b2c {

// Two-state splats (--cage-open): a bound splat carries a second colour and an opacity delta, its "open" state, blended
// in by a GEOMETRIC gate computed from the cage alone.
//
// Why: one static colour per splat cannot be both the closed, shaded crease of the capture (the A-pose armpit) and the
// lit skin of a raised-arm pose; the appearance MLP (gpu/cage_app.h) could, but its learned trigger extrapolates badly.
// Here the trigger is fixed geometry and only the two states are learned: the capture views teach the base splat (they
// see g = 0), the posed clips teach t and dO (their gradient reaches the base scaled by 1 - g). A splat whose open pose
// looks like its closed one (a garment under the arm) learns t = its own colour and is left alone: no segmentation decides.
//
// The gate (per cage vertex, splats interpolate it by their barycentrics): theta, how far a joint next to the vertex
// has turned (degrees), per cage frame. It is a decision about the rig, so b2crig computes it (b2crig/rig/open_gate.py:
// the largest rotation, relative to the vertex's own triangle frame, of the cage vertices near it in the canonical
// pose) and ships it in the cage file's B2COPEN1 section (gpu/cage.h); a cage without it cannot drive open states.
// g = clamp((theta - start) / (end - start), 0, 1) in degrees; the SH colour c becomes c + g d (d a per-splat colour
// OFFSET, so the view-dependent SH detail survives where the gate is fully on; a blend towards a flat target colour
// dulled a metallic knee pad) and the posed opacity logit gains g * dO (dO <= max_do, 0 by default: the open state may
// only fade a splat). The base colour receives (1 - g) of the posed views' gradient, the offset g of it.
//
// The per-splat parameters live next to the model's (Adam, lazy: a splat steps only when it received a gradient),
// follow refine (a child copies its parent's state) and export as ply properties open_r/g/b (the offset, -1..1) and
// open_dopacity; the gate's parameters go into the header comment "b2c.cage_open add START END MAXDO RADIUS MINDIST"
// so render --cage maps theta the same way without flags (older headers carry two more fields, b2crig's radius and
// min_dist, which are ignored).

struct CageOpen {
  bool on = false;
  int cap = 0;                        // allocated splats
  int nv = 0, nf = 0;
  float start = 30.f, end = 70.f, max_do = 0.f;   // degrees
  float reg_col = 0.f;                // training: pull of the colour offset towards 0, added to dL/dd (W d)
  // Consistency gate of the update (training): the open state of a splat steps only as far as its gradient has been
  // CONSISTENT: the Adam ratio r = m_hat / sqrt(v_hat) (1 for a steady gradient, ~sqrt((1 - b1) / (1 + b1)) for noise)
  // enters as sign(r) max(|r| - snr, 0) / (1 - snr). An opened crease pushes every frame the same way; a knee pad
  // misaligned by a few pixels between WAN and the render pushes each splat in a different direction every frame and
  // used to end up as rainbow speckles. beta1 is raised for the open state so the ratio separates the two better.
  float snr = 0.f, beta1 = 0.9f;
  bool stats_grad = false;            // log the data-gradient magnitudes at the next step()
  DevBuf<float> gvert;                // [nv] g of the current frame
  // per-splat state
  DevBuf<float4> P, mP, vP;           // [cap]: (offset r, g, b, dO) and Adam moments
  DevBuf<float> gsplat;               // [cap]: g of the current frame
  DevBuf<float4> splat_app;           // (offset rgb, g): RenderParams::app with app_add
  DevBuf<float4> col_base;            // the SH colour before the blend (RenderParams::col_base)
  DevBuf<float4> g_col;               // (dL/d offset rgb, unused)
  DevBuf<float2> g_app;               // (dL/d posed opacity logit, dL/d posed log scale): OptimParams::g_app_out
  int adam_t = 0;
  float stat_g = 0.f, stat_do = 0.f; int stat_open = 0;   // last apply with stats: mean g / |dO| over splats, splats with g > 0

  // Open states from the cloud (open_* when trained before, else 0). The cage must carry its B2COPEN1 section.
  void init(const CageRig& cage, const SplatCloud& c, cudaStream_t stream);
  void ensure(int want, cudaStream_t stream);   // capacity for `want` splats
  // After cage.pose(): g per vertex (from the posed frame's theta) and splat, splat_app, and g * dO added to cage.pos_view's opacity logit.
  void apply(const CageRig& cage, const Model& m, cudaStream_t stream, bool stats = false);
  // After the rasteriser backward, before the optimiser: dL/dt from ctx.v_splat's colour lanes, which are then scaled by (1 - g).
  void pre_optim(RenderCtx& ctx, const Model& m, cudaStream_t stream);
  // After the optimiser: Adam on the open states of the splats that received a gradient.
  void step(const Model& m, float lr_col, float lr_do, cudaStream_t stream);
  // After refine: child k takes parent k's state.
  void inherit(const uint32_t* parents, const uint32_t* children, uint32_t pairs, int n, cudaStream_t stream);
  // Render diagnostics: blend every splat towards magenta by g.
  void debug_g(int n, cudaStream_t stream);
  // Open states of the first c.n splats into the cloud (has_open set).
  void download(SplatCloud& c, cudaStream_t stream) const;
  std::string header_comment() const;
  bool parse_header(const std::vector<std::string>& comments);   // sets the gate parameters from it; false when absent
};

}  // namespace b2c
