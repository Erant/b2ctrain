#include "gpu/render.h"
#include "gpu/splat_math.cuh"
#include <cub/cub.cuh>

namespace b2c {

namespace {

// Depth keys for the per-splat sort: raw float bits of view-space z (monotone for z > 0); culled -> max.
__global__ void depth_keys_kernel(int n, const uint32_t* __restrict__ tile_count, const float4* __restrict__ proj1, uint32_t* __restrict__ keys, uint32_t* __restrict__ vals) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  keys[i] = tile_count[i] ? __float_as_uint(proj1[i].z) : 0xFFFFFFFFu;
  vals[i] = (uint32_t)i;
}
__global__ void permute_counts_kernel(int n, const uint32_t* __restrict__ order, const uint32_t* __restrict__ tile_count, uint32_t* __restrict__ out) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < n) out[r] = tile_count[order[r]];
}

// Emits intersections in depth order (thread r handles the r-th nearest splat); keys carry the tile id only.
// Emits intersections in depth order with coalesced writes: the 32 splats of a warp own a contiguous slot range
// [start, end); lanes sweep it 32 slots at a time, each slot mapping to (owning splat, n-th hit tile of its mask).
// Splats whose bbox exceeds 32 tiles (mask sentinel) are handled per thread afterwards.
__global__ void __launch_bounds__(256) emit_kernel(int n, const uint32_t* __restrict__ order, const uint32_t* __restrict__ tile_count, const uint32_t* __restrict__ tile_off_incl, const uint2* __restrict__ hit_info,
                            const float4* __restrict__ proj0, const float4* __restrict__ proj1,
                            int tiles_x, int tiles_y, uint32_t cap,
                            uint32_t* __restrict__ keys, uint32_t* __restrict__ vals) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane = threadIdx.x & 31;
  bool valid = r < n;
  int i = valid ? (int)order[r] : 0;
  uint32_t cnt = valid ? tile_count[i] : 0u;
  uint32_t end = valid ? tile_off_incl[r] : 0u;
  uint32_t base = end - cnt;
  uint2 hi = valid && cnt ? hit_info[i] : make_uint2(0u, 0u);
  bool big = cnt && hi.x == 0xFFFFFFFFu;
  if (big) cnt = 0;  // handled below; exclude from the cooperative sweep
  uint32_t my_start = valid ? base : 0xFFFFFFFFu, my_cnt = cnt;
  if (base + cnt > cap) my_cnt = 0;  // overflow: host retries with a bigger buffer
  const uint32_t sentinel_tile = (uint32_t)(tiles_x * tiles_y);
  // Warp range.
  unsigned active = __ballot_sync(0xffffffffu, my_cnt > 0);
  if (active) {
    int first = __ffs(active) - 1, last = 31 - __clz(active);
    uint32_t wstart = __shfl_sync(0xffffffffu, my_start, first);
    uint32_t wend = __shfl_sync(0xffffffffu, my_start + my_cnt, last);
    int min_x = (int)(hi.y & 0xFFFu), min_y = (int)((hi.y >> 12) & 0xFFFu), bw = (int)(hi.y >> 24);
    for (uint32_t sb = wstart; sb < wend; sb += 32) {  // warp-uniform trip count
      uint32_t s = sb + lane;
      bool in_range = s < wend;
      // Owner = the largest lane whose start <= s (starts are sorted across lanes; empty lanes share the next start,
      // invalid lanes sit at 0xFFFFFFFF). Binary search with shuffles.
      int lo_l = 0, hi_l = 31;
#pragma unroll
      for (int step = 0; step < 5; step++) {
        int mid = (lo_l + hi_l + 1) >> 1;
        uint32_t st = __shfl_sync(0xffffffffu, my_start, mid);
        if (st <= s) lo_l = mid; else hi_l = mid - 1;
      }
      int o = lo_l;
      uint32_t o_start = __shfl_sync(0xffffffffu, my_start, o);
      uint32_t o_cnt = __shfl_sync(0xffffffffu, my_cnt, o);
      bool has_owner = o_cnt > 0 && s < o_start + o_cnt;  // false for slots of big splats (written below)
      uint32_t o_mask = __shfl_sync(0xffffffffu, hi.x, o);
      int o_minx = __shfl_sync(0xffffffffu, min_x, o), o_miny = __shfl_sync(0xffffffffu, min_y, o), o_bw = __shfl_sync(0xffffffffu, bw, o);
      int o_i = __shfl_sync(0xffffffffu, i, o);
      int k = (int)(s - o_start);
      int bit = __fns(o_mask, 0, k + 1);
      uint32_t key = sentinel_tile;
      if (bit >= 0 && bit < 32) { int tx = o_minx + bit % o_bw, ty = o_miny + bit / o_bw; key = (uint32_t)(ty * tiles_x + tx); }
      if (in_range && has_owner) { keys[s] = key; vals[s] = (uint32_t)o_i; }
    }
  }
  if (big) {
    uint32_t bcnt = end - base;
    if (base + bcnt > cap) return;
    float4 a = proj0[i], b = proj1[i];
    Sym2 conic{a.z, a.w, b.x};
    float power = b.w;
    float detc = conic.c00 * conic.c11 - conic.c01 * conic.c01;
    float inv_detc = 1.f / detc;
    float ex = sqrtf(2.f * power * conic.c11 * inv_detc), ey = sqrtf(2.f * power * conic.c00 * inv_detc);
    TileBox bb = tile_bbox(a.x, a.y, ex, ey, tiles_x, tiles_y);
    uint32_t written = 0;
    for (int ty = bb.min_y; ty < bb.max_y && written < bcnt; ty++)
      for (int tx = bb.min_x; tx < bb.max_x && written < bcnt; tx++) {
        float rx = tx * (float)RT_W, ry = ty * (float)RT_W;
        if (tile_hit(rx, ry, rx + RT_W, ry + RT_W, a.x, a.y, conic, power)) { keys[base + written] = (uint32_t)(ty * tiles_x + tx); vals[base + written] = (uint32_t)i; written++; }
      }
    for (; written < bcnt; written++) { keys[base + written] = sentinel_tile; vals[base + written] = (uint32_t)i; }
  }
}

__global__ void tile_ranges_kernel(uint32_t n, const uint32_t* __restrict__ keys, uint32_t n_tiles, uint2* __restrict__ ranges) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t tile = keys[i];
  uint32_t prev = i == 0 ? 0xFFFFFFFFu : keys[i - 1];
  if (i == 0 || tile != prev) {
    if (tile < n_tiles) ranges[tile].x = i;
    if (i > 0 && prev < n_tiles) ranges[prev].y = i;
  }
  if (i == n - 1 && tile < n_tiles) ranges[tile].y = n;
}

}  // namespace

void bin_and_sort(RenderCtx& ctx, const Model& m, cudaStream_t stream) {
  if (m.n == 0) { ctx.num_isect = 0; ctx.tile_ranges.zero(stream); return; }
  int n = m.n;
  size_t tmp = 0;
  // 1. Depth-sort the splats (culled ones sort last).
  depth_keys_kernel<<<div_up(n, PROJ_BLOCK), PROJ_BLOCK, 0, stream>>>(n, ctx.tile_count, ctx.proj1, ctx.depth_keys, ctx.order_in);
  CUDA_KERNEL_CHECK();
  cub::DeviceRadixSort::SortPairs(nullptr, tmp, ctx.depth_keys.ptr, ctx.depth_keys_sorted.ptr, ctx.order_in.ptr, ctx.order.ptr, n, 0, 32, stream);
  ctx.cub_tmp.reserve(tmp);
  cub::DeviceRadixSort::SortPairs(ctx.cub_tmp.ptr, tmp, ctx.depth_keys.ptr, ctx.depth_keys_sorted.ptr, ctx.order_in.ptr, ctx.order.ptr, n, 0, 32, stream);
  // 2. Scan of tile counts in depth order.
  permute_counts_kernel<<<div_up(n, PROJ_BLOCK), PROJ_BLOCK, 0, stream>>>(n, ctx.order, ctx.tile_count, ctx.count_perm);
  CUDA_KERNEL_CHECK();
  cub::DeviceScan::InclusiveSum(nullptr, tmp, ctx.count_perm.ptr, ctx.tile_off.ptr, n, stream);
  ctx.cub_tmp.reserve(tmp);
  cub::DeviceScan::InclusiveSum(ctx.cub_tmp.ptr, tmp, ctx.count_perm.ptr, ctx.tile_off.ptr, n, stream);
  CUDA_CHECK(cudaMemcpyAsync(ctx.h_count.ptr, ctx.tile_off.ptr + (n - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  uint32_t total = ctx.h_count.ptr[0];
  ctx.num_isect = total;
  if (total > ctx.isect_cap) {
    size_t ncap = ctx.isect_cap; while (ncap < total) ncap *= 2;
    ctx.keys.reserve(ncap); ctx.vals.reserve(ncap); ctx.keys_sorted.reserve(ncap); ctx.vals_sorted.reserve(ncap);
    ctx.isect_cap = ncap;
  }
  ctx.tile_ranges.zero(stream);
  if (total == 0) return;
  // 3. Emit in depth order, then a stable sort by tile id alone keeps the depth order within each tile.
  emit_kernel<<<div_up(n, PROJ_BLOCK), PROJ_BLOCK, 0, stream>>>(n, ctx.order, ctx.tile_count, ctx.tile_off, ctx.hit_info, ctx.proj0, ctx.proj1, ctx.tiles_x, ctx.tiles_y, (uint32_t)ctx.isect_cap, ctx.keys, ctx.vals);
  CUDA_KERNEL_CHECK();
  size_t tmp2 = 0;
  cub::DeviceRadixSort::SortPairs(nullptr, tmp2, ctx.keys.ptr, ctx.keys_sorted.ptr, ctx.vals.ptr, ctx.vals_sorted.ptr, (int)total, 0, ctx.tile_bits, stream);
  ctx.cub_tmp.reserve(tmp2);
  cub::DeviceRadixSort::SortPairs(ctx.cub_tmp.ptr, tmp2, ctx.keys.ptr, ctx.keys_sorted.ptr, ctx.vals.ptr, ctx.vals_sorted.ptr, (int)total, 0, ctx.tile_bits, stream);
  tile_ranges_kernel<<<div_up(total, 256), 256, 0, stream>>>(total, ctx.keys_sorted, (uint32_t)ctx.n_tiles, ctx.tile_ranges);
  CUDA_KERNEL_CHECK();
}

}  // namespace b2c
