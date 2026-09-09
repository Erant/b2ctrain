#pragma once
#include "gpu/util.cuh"
#include "model.h"
#include "dataset/camera.h"

namespace b2c {

struct CameraGPU {
  float R[9];          // world-to-camera rotation, row-major
  float t[3];
  float pos[3];        // camera center (world)
  float fx, fy, cx, cy;
  int W, H;
  float lim_pos_x, lim_pos_y, lim_neg_x, lim_neg_y;  // EWA Jacobian clamp
  static CameraGPU from(const Camera& c, int W, int H) {
    CameraGPU g;
    for (int i = 0; i < 9; i++) g.R[i] = c.R[i];
    for (int i = 0; i < 3; i++) { g.t[i] = c.t[i]; g.pos[i] = c.pos[i]; }
    float sx = (float)W / c.width, sy = (float)H / c.height;
    g.fx = c.fx * sx; g.fy = c.fy * sy; g.cx = c.cx * sx; g.cy = c.cy * sy; g.W = W; g.H = H;
    g.lim_pos_x = (1.15f * W - g.cx) / g.fx; g.lim_pos_y = (1.15f * H - g.cy) / g.fy;
    g.lim_neg_x = (-0.15f * W - g.cx) / g.fx; g.lim_neg_y = (-0.15f * H - g.cy) / g.fy;
    return g;
  }
};

enum class FeatureMode { None, Normals, Buffer };

struct RenderParams {
  CameraGPU cam;
  float bg[3] = {0, 0, 0};
  int sh_degree = 3;            // active degree (<= model degree)
  FeatureMode feat = FeatureMode::None;
  const float* feat_buffer = nullptr;  // [n][3] when feat == Buffer
  bool mip = false;
  bool bwd_info = true;         // write per-pixel last index / shrink tile ranges
  // Render-side alignment (--align-warp render): a per-view displacement field on a warp_w x warp_h grid covering the
  // image (full-resolution pixel units, the same smoothed and capped flow the frame warp would apply). Each splat's
  // projected mean is moved by minus the field sampled there, so the render meets the pristine frame where the frame
  // disagrees with the model, instead of the frame being resampled towards the render. A constant local shift, so
  // the gradient of the mean is unchanged.
  const float2* warp = nullptr;
  int warp_w = 0, warp_h = 0;
  // Articulated per-view deformation (gpu/deform.h): render from these means instead of the model's canonical ones.
  const float4* pos_override = nullptr;
  // Hollow loss: per-pixel reference depth of the body surface (camera z, +inf = no surface). A fragment at depth z
  // is penalised by h(z) = clamp((z - z_ref - margin) / margin, 0, 1) times its compositing weight; the gradient of
  // that weight runs through every fragment in front of it, which is what pushes the surface opaque.
  const float* hollow_z = nullptr;
  float hollow_margin = 0.05f;
  float hollow_lam = 0.f;       // dL/d(penalised weight) per pixel, before grad_scale
};

// Persistent scratch for rendering one view; grows on demand.
struct RenderCtx {
  int W = 0, H = 0, tiles_x = 0, tiles_y = 0, n_tiles = 0, tile_bits = 0, depth_bits = 0;
  // per-splat (global index)
  DevBuf<float4> proj0, proj1, proj2;   // (xy, c00, c01) | (c11, opac, depth, power = ln(255 opac)) | (r, g, b, feat_x)
  DevBuf<float2> proj3;                 // (feat_y, feat_z)
  DevBuf<uint32_t> tile_count, tile_off;  // per-splat hit count, inclusive scan (in depth order)
  DevBuf<uint2> hit_info;                 // per-splat (bitmask of hit tiles within the bbox, packed bbox min_x | min_y << 12 | bw << 24); mask 0xFFFFFFFF = recompute
  DevBuf<uint32_t> depth_keys, depth_keys_sorted, order_in, order;  // per-splat depth sort
  DevBuf<uint32_t> count_perm;            // tile_count permuted into depth order
  // intersections
  DevBuf<uint32_t> keys, vals, keys_sorted, vals_sorted;
  size_t isect_cap = 0;
  uint32_t num_isect = 0;
  DevBuf<unsigned char> cub_tmp;
  PinnedBuf<uint32_t> h_count;
  // per-tile / per-pixel
  DevBuf<uint2> tile_ranges;
  DevBuf<float4> out_rgba;    // final rgb (with bg) + alpha
  DevBuf<float4> out_feat;    // feat xyz + final transmittance
  DevBuf<float> hollow_pen;   // per-pixel sum of weight * h(depth) when the hollow loss is on
  DevBuf<uint32_t> last_idx;  // one past the last contributing intersection per pixel
  // backward
  DevBuf<float> v_splat;      // [n][13] per-splat 2D gradients: xy(2) conic(3) rgb(3) opac(1) refine(1) feat(3)
  DevBuf<uint32_t> vis_flag;  // [n] set when a splat contributed to any pixel this step
  DevBuf<float4> v_out;       // dL/d(rgba) per pixel
  DevBuf<float4> v_feat;      // dL/d(feat) per pixel
  DevBuf<float> loss_accum;   // [8] scalar accumulators

  void setup(int W, int H, int n_splats, cudaStream_t stream);
};

// Full forward: projection, binning, sort, rasterisation. Leaves outputs in ctx.
void render_forward(RenderCtx& ctx, const Model& m, const RenderParams& p, cudaStream_t stream);

// Individual stages (exposed for tests / evidence).
void project_splats(RenderCtx& ctx, const Model& m, const RenderParams& p, cudaStream_t stream);
void bin_and_sort(RenderCtx& ctx, const Model& m, cudaStream_t stream);
void rasterize_forward(RenderCtx& ctx, const RenderParams& p, cudaStream_t stream);

// Backward through rasterisation: consumes ctx.v_out / ctx.v_feat, produces ctx.v_splat; marks vis_count in the model.
void rasterize_backward(RenderCtx& ctx, const Model& m, const RenderParams& p, cudaStream_t stream);
// Tensor-core variant. The per-pixel gradients (ctx.v_out / v_feat) must have been produced with `grad_scale` applied;
// the kernel divides the accumulated per-splat gradients by it.
void rasterize_backward_tc(RenderCtx& ctx, const Model& m, const RenderParams& p, float grad_scale, cudaStream_t stream);

}  // namespace b2c
