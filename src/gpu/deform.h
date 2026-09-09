#pragma once
#include "gpu/util.cuh"
#include "model.h"
#include <string>
#include <vector>

namespace b2c {

// Per-view articulated deformation of the canonical splat (the experiment of 2026-09-09: the generated frames move
// the arms by centimetres between segments of the orbit, and a single canonical body cannot explain them all).
//
// A rig file (b2crunner's body tooling, from the refit MHR body) carries a subsample of the body's vertices in the
// canonical (refit) pose with their top-4 skinning joints and weights, the joint tree with the canonical joint
// positions, and the set of ACTIVE joints (the arm chains). Per training view and active joint the trainer learns a
// small rotation about that joint's pivot; forward kinematics composes them top-down into one rigid transform per
// joint, every splat is bound to its nearest rig vertex (re-bound after every refine) and rendered for view v at the
// linear blend of its joints' transforms. The gradient of a joint's rotation is the torque of the splats' positional
// gradients about its (moved) pivot, summed over the joint's subtree; Adam per (view, joint) with an optional
// smoothness pull towards the neighbouring views. The model itself stays canonical and is what gets exported.
struct BodyRig {
  static constexpr int MAX_ANC = 16;   // active ancestors (self included) a joint can have (a finger chain is ~12 deep)
  int nv = 0, nj = 0, nviews = 0, n_active = 0;
  std::vector<std::string> names;      // view names (frame file names)
  std::vector<int> parents_h, active_h;
  DevBuf<float3> verts;                // [nv] canonical positions, world
  DevBuf<int4> vj; DevBuf<float4> vw;  // [nv] joints and weights
  DevBuf<int> parents, active;         // [nj]; active: 1 when the joint carries a per-view rotation
  DevBuf<int> anc;                     // [nj][MAX_ANC] active ancestors-or-self, -1 padded
  DevBuf<float3> jpos0;                // [nj] canonical joint positions, world
  DevBuf<float> xf;                    // [nviews][nj][12]: R row-major (9), t (3), written by fk()
  DevBuf<float3> pivots;               // [nj] the current view's moved pivots (fk scratch)
  // learnable per-view rotations (axis-angle, world axes) and their Adam moments, [nviews][nj]
  DevBuf<float3> omega, m_om, v_om;
  DevBuf<float3> torque;               // [nj] scratch
  DevBuf<float3> g_pos;                // [cap] per-splat dL/d(posed mean), filled by the optimizer
  int adam_t = 0;
  // per-splat binding and the posed positions for the current view
  DevBuf<int4> bind_j; DevBuf<float4> bind_w;
  DevBuf<float4> pos_view;             // (x, y, z, opacity logit) as pos_op
  int bound_n = 0;

  bool load(const std::string& path);  // false when the file is missing; throws on a malformed one
  int view_index(const std::string& name) const;
  // Nearest-vertex binding of `n` points with `stride` floats per point (4 for pos_op, 3 for mesh vertices).
  void bind_points(const float* pos, int n, int stride, DevBuf<int4>& bj, DevBuf<float4>& bw, cudaStream_t stream) const;
  void bind(const Model& m, cudaStream_t stream);
  // Forward kinematics of view `v` from its rotations into xf and pivots.
  void fk(int v, cudaStream_t stream);
  // fk(v), then pose the model's means into pos_view.
  void pose(int v, const Model& m, cudaStream_t stream);
  // Pose `n` float3 points (bound with bind_points) for view `v` with the transforms fk() last wrote for it.
  void pose_points(int v, const float3* src, const int4* bj, const float4* bw, int n, float3* dst, cudaStream_t stream) const;
  // After the optimizer wrote g_pos for the step rendered from pose(v): torque per active joint, then an Adam step on
  // the view's rotations with `lr` (radians) and a pull of `smooth` towards the mean of the two neighbouring views.
  // `zero`: pull towards no rotation; `global_v`: one second moment for all joints and views (evidence-weighted steps).
  void update(int v, const Model& m, float lr, float smooth, float zero, bool global_v, cudaStream_t stream);
  // Host copy of omega [nviews][nj] and a summary for the log.
  std::vector<float3> download_omega(cudaStream_t stream) const;
  float mean_abs_torque = 0.f;          // last update's mean |torque| over active joints (host, when sampled)
  void sample_torque(cudaStream_t stream);
};

}  // namespace b2c
