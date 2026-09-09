#include "gpu/render.h"

namespace b2c {

void RenderCtx::setup(int w, int h, int n, cudaStream_t stream) {
  W = w; H = h; tiles_x = (W + RT_W - 1) / RT_W; tiles_y = (H + RT_W - 1) / RT_W; n_tiles = tiles_x * tiles_y;
  tile_bits = 1; while ((1 << tile_bits) < n_tiles + 1) tile_bits++;
  depth_bits = 32 - tile_bits; if (depth_bits > 20) depth_bits = 20;
  size_t nn = (size_t)n;
  proj0.reserve(nn); proj1.reserve(nn); proj2.reserve(nn); proj3.reserve(nn);
  tile_count.reserve(nn); tile_off.reserve(nn); hit_info.reserve(nn);
  depth_keys.reserve(nn); depth_keys_sorted.reserve(nn); order_in.reserve(nn); order.reserve(nn); count_perm.reserve(nn);
  if (v_splat.count < nn * 13) { v_splat.reserve(nn * 13); v_splat.zero(stream); }
  if (vis_flag.count < nn) { vis_flag.reserve(nn); vis_flag.zero(stream); }
  tile_ranges.reserve(n_tiles);
  size_t npx = (size_t)W * H;
  out_rgba.reserve(npx); out_feat.reserve(npx); hollow_pen.reserve(npx); last_idx.reserve(npx); v_out.reserve(npx); v_feat.reserve(npx);
  loss_accum.reserve(16);
  h_count.reserve(4);
  if (isect_cap == 0) { isect_cap = 1u << 22; keys.reserve(isect_cap); vals.reserve(isect_cap); keys_sorted.reserve(isect_cap); vals_sorted.reserve(isect_cap); }
}

void render_forward(RenderCtx& ctx, const Model& m, const RenderParams& p, cudaStream_t stream) {
  project_splats(ctx, m, p, stream);
  bin_and_sort(ctx, m, stream);
  rasterize_forward(ctx, p, stream);
}

}  // namespace b2c
