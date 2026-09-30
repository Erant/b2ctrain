#pragma once
#include "gpu/cage.h"
#include "gpu/render.h"
#include <string>

namespace b2c {

// Pose-dependent appearance of a cage-posed splat (--cage-app): a small MLP evaluated per cage VERTEX from the local
// stretch of the surface around it, whose outputs every bound splat interpolates by its barycentrics.
//
// Why: one static colour / opacity per splat cannot be both the closed, shaded crease of the capture (the A-pose
// armpit) and the opened, lit surface of a raised-arm pose; LBS stretches the crease into a membrane whose dark splats
// read as a web. The MLP learns what an opening (or closing) crease looks like from the posed training clips.
//
// Inputs per vertex: 6 stretch features and a learnable latent z (APP_LAT). The features are the log principal
// stretches (log s1 >= log s2) of the 2D deformation gradient of the incident cage triangles (canonical -> posed),
// canonical-area-weighted at the vertex, then again blurred over ~2 and ~6 rings (face mean <-> vertex mean steps):
// all zero at the canonical pose. Two more measure the change in local OCCLUSION: log((n + 1) / (n0 + 1)) with n the
// number of cage vertices within 4 cm / 8 cm of the vertex in the posed frame, n0 in the canonical one. An opening crease
// exposes surface that does not stretch (the inner arm and the side of the torso as the arm lifts, and the splats
// that sat hidden between them): its neighbour count drops; a closing elbow's rises. Outputs R (APP_OUT) are the MLP at the features MINUS the MLP at zero features
// (same z) for the blend, opacity and scale channels, so the canonical pose renders the plain splat exactly:
//   alpha = tanh(R0)            signed blend weight of the splat colour towards the target t (c' = c + alpha (t - c))
//   t     = sigmoid(R1..3)      target colour (not rest-subtracted; it only acts through alpha)
//   dO    = min(R4, max_do)     added to the posed opacity logit
//   dS    = max_ds tanh(R5 / max_ds)   added to the posed log scales (isotropic; max_ds 0 = off)
//   dP    = max_dp tanh(R6..8 / max_dp) metres, in the posed triangle frame (e1, e2, n), added to the posed mean
//           (max_dp 0 = off): lets the membrane's splats move back onto the arm / torso
// Dead zone: every feature f enters as sign(f) max(|f| - dz, 0), so the surface must stretch (or compress) past
// exp(dz) before the MLP sees it: most of the body deforms a little in any pose, and without the dead zone the MLP
// learns whatever differs between posed and canonical training views (the video's look vs the capture's) instead
// of the creases.
// Being per vertex, the model does not care which splats exist: refine, prune and compaction leave it valid.
//
// Training: the per-splat gradients (colour lanes of the rasteriser backward; opacity-logit and log-scale gradients
// from the optimiser, OptimParams::g_app_out) are chained through the activations, scattered to the three vertices
// of the splat's triangle and back-propagated through the MLP (weights and latents, Adam). The base colour's gradient
// is scaled by (1 - alpha): where the target replaces the colour, the view does not teach the base splat.
//
// File (<export>.app): char magic[8] = "B2CAPP03"; int32 nv, n_feat, n_lat, n_hid, n_out; float dz, max_do, max_ds, max_dp;
// float params[APP_NP]; float z[nv][n_lat].
constexpr int APP_FEAT = 8, APP_LAT = 8, APP_IN = APP_FEAT + APP_LAT, APP_HID = 32, APP_OUT = 9;
constexpr int APP_NP = APP_HID * APP_IN + APP_HID + APP_HID * APP_HID + APP_HID + APP_OUT * APP_HID + APP_OUT;

struct CageApp {
  int nv = 0, nf = 0;
  bool on = false;
  float dz = 0.f, max_do = 1e30f, max_ds = 1.f, max_dp = 0.f;
  // training-only regularisers: rise penalties (added to dL/d dO, dL/d dS of every visible splat whose opacity / scale
  // the MLP raises), L2 on the vertex outputs, decoupled weight decay of the latents
  float reg_rise_do = 0.f, reg_rise_ds = 0.f, reg_out = 0.f, latent_decay = 0.f;
  bool stats_grad = false;             // log the data-gradient magnitudes at the next step()   // dead zone, output limits (see above; saved with the model)
  // topology (canonical, from the cage)
  DevBuf<int> vf_off, vf_idx;          // vertex -> incident faces (CSR)
  DevBuf<float4> tri_inv0;             // [nf] inverse of the canonical edge matrix in the canonical triangle frame
  DevBuf<float> tri_area0, vert_area0; // canonical face area, per-vertex sum of incident face areas
  DevBuf<float2> vcnt0;                // canonical neighbour counts (4 cm, 8 cm)
  // per frame
  DevBuf<float2> ffeat, vtmp;          // scratch: per face, per vertex
  DevBuf<float> vfeat;                 // [nv][APP_FEAT]
  DevBuf<float> vout;                  // [nv][APP_OUT] R
  DevBuf<float> g_vout;                // [nv][APP_OUT] dL/dR
  // parameters
  DevBuf<float> P, mP, vP, gP;         // [APP_NP]
  DevBuf<float> Z, mZ, vZ, gZ;         // [nv][APP_LAT]
  int adam_t = 0;
  // per splat (current frame)
  DevBuf<float4> splat_app;            // (t.rgb, alpha): RenderParams::app
  DevBuf<float4> col_base;             // the SH colour before the blend (RenderParams::col_base)
  DevBuf<float> splat_ds;              // tanh of the scale channel
  DevBuf<float4> splat_dp;             // tanh of the position channels
  DevBuf<float3> g_pos;                // dL/d posed mean (OptimParams::g_pos_out)
  DevBuf<float4> g_col;                // (dL/dt.rgb, dL/dalpha)
  DevBuf<float2> g_app;                // (dL/d posed opacity logit, dL/d posed log scale summed over axes): OptimParams
  // statistics of the last apply (host, filled when stats are requested)
  float stat_alpha = 0.f, stat_do = 0.f, stat_ds = 0.f, stat_dp = 0.f;

  // Topology from the cage (after CageRig::load); random init of the MLP (weights) and latents.
  void init(const CageRig& cage, uint32_t seed, cudaStream_t stream);
  bool load(const std::string& path, const CageRig& cage, cudaStream_t stream);   // false when missing; throws on mismatch
  void save(const std::string& path, cudaStream_t stream) const;
  // After cage.pose(): features and MLP of the posed frame, then the per-splat blend (splat_app) and the opacity /
  // scale deltas applied in place to cage.pos_view / cage.lscale_view.
  void apply(CageRig& cage, const Model& m, cudaStream_t stream, bool stats = false);
  // After the rasteriser backward, before the optimiser: blend gradients from ctx.v_splat's colour lanes, which are
  // then scaled by (1 - alpha); zeroes g_app for the optimiser to fill.
  void pre_optim(RenderCtx& ctx, const Model& m, cudaStream_t stream);
  // The per-vertex halves (exposed for the gradient test): features + MLP of cage.verts_view into vout; g_vout
  // back-propagated into gP / gZ (accumulated; g_vout is zeroed).
  void vert_forward(const CageRig& cage, cudaStream_t stream);
  void vert_backward(cudaStream_t stream);
  // Render diagnostics: blend every splat towards magenta by |alpha| (where and how strongly the MLP recolours).
  void debug_alpha(int n, cudaStream_t stream);
  // After the optimiser: chain everything to the vertices, back-propagate, Adam on weights and latents.
  void step(const CageRig& cage, const Model& m, float lr, float lr_lat, cudaStream_t stream);
};

}  // namespace b2c
