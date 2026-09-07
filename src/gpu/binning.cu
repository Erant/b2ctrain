#include "gpu/render.h"
#include "gpu/splat_math.cuh"
#include <cub/cub.cuh>

namespace b2c {

namespace {

__global__ void emit_kernel(int n, const uint32_t* __restrict__ tile_count, const uint32_t* __restrict__ tile_off_incl,
                            const float4* __restrict__ proj0, const float4* __restrict__ proj1,
                            int tiles_x, int tiles_y, int depth_bits, uint32_t cap,
                            uint32_t* __restrict__ keys, uint32_t* __restrict__ vals) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t cnt = tile_count[i];
  if (cnt == 0) return;
  uint32_t base = tile_off_incl[i] - cnt;
  if (base + cnt > cap) return;  // overflow: host detects via total and retries with a bigger buffer
  float4 a = proj0[i], b = proj1[i];
  Sym2 conic{a.z, a.w, b.x};
  float power = __logf(b.y * 255.f);
  float detc = conic.c00 * conic.c11 - conic.c01 * conic.c01;
  float inv_detc = 1.f / detc;
  float ex = sqrtf(2.f * power * conic.c11 * inv_detc), ey = sqrtf(2.f * power * conic.c00 * inv_detc);
  TileBox bb = tile_bbox(a.x, a.y, ex, ey, tiles_x, tiles_y);
  uint32_t dcode = depth_code(b.z, depth_bits);
  uint32_t written = 0;
  uint32_t sentinel_tile = (uint32_t)(tiles_x * tiles_y);
  for (int ty = bb.min_y; ty < bb.max_y && written < cnt; ty++)
    for (int tx = bb.min_x; tx < bb.max_x && written < cnt; tx++) {
      float rx = tx * (float)TILE_W, ry = ty * (float)TILE_W;
      if (tile_hit(rx, ry, rx + TILE_W, ry + TILE_W, a.x, a.y, conic, power)) {
        uint32_t tile = (uint32_t)(ty * tiles_x + tx);
        keys[base + written] = (tile << depth_bits) | dcode;
        vals[base + written] = (uint32_t)i;
        written++;
      }
    }
  // Pad any slots the count pass promised but this pass did not fill (FP drift guard).
  for (; written < cnt; written++) { keys[base + written] = sentinel_tile << depth_bits; vals[base + written] = (uint32_t)i; }
}

__global__ void tile_ranges_kernel(uint32_t n, const uint32_t* __restrict__ keys, int depth_bits, uint32_t n_tiles, uint2* __restrict__ ranges) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t tile = keys[i] >> depth_bits;
  uint32_t prev = i == 0 ? 0xFFFFFFFFu : keys[i - 1] >> depth_bits;
  if (i == 0 || tile != prev) {
    if (tile < n_tiles) ranges[tile].x = i;
    if (i > 0 && prev < n_tiles) ranges[prev].y = i;
  }
  if (i == n - 1 && tile < n_tiles) ranges[tile].y = n;
}

}  // namespace

void bin_and_sort(RenderCtx& ctx, const Model& m, cudaStream_t stream) {
  if (m.n == 0) { ctx.num_isect = 0; ctx.tile_ranges.zero(stream); return; }
  // Inclusive scan of tile counts.
  size_t tmp = 0;
  cub::DeviceScan::InclusiveSum(nullptr, tmp, ctx.tile_count.ptr, ctx.tile_off.ptr, m.n, stream);
  ctx.cub_tmp.reserve(tmp);
  cub::DeviceScan::InclusiveSum(ctx.cub_tmp.ptr, tmp, ctx.tile_count.ptr, ctx.tile_off.ptr, m.n, stream);
  CUDA_CHECK(cudaMemcpyAsync(ctx.h_count.ptr, ctx.tile_off.ptr + (m.n - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
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
  emit_kernel<<<div_up(m.n, PROJ_BLOCK), PROJ_BLOCK, 0, stream>>>(m.n, ctx.tile_count, ctx.tile_off, ctx.proj0, ctx.proj1, ctx.tiles_x, ctx.tiles_y, ctx.depth_bits, (uint32_t)ctx.isect_cap, ctx.keys, ctx.vals);
  CUDA_KERNEL_CHECK();
  int end_bit = ctx.depth_bits + ctx.tile_bits; if (end_bit > 32) end_bit = 32;
  size_t tmp2 = 0;
  cub::DeviceRadixSort::SortPairs(nullptr, tmp2, ctx.keys.ptr, ctx.keys_sorted.ptr, ctx.vals.ptr, ctx.vals_sorted.ptr, (int)total, 0, end_bit, stream);
  ctx.cub_tmp.reserve(tmp2);
  cub::DeviceRadixSort::SortPairs(ctx.cub_tmp.ptr, tmp2, ctx.keys.ptr, ctx.keys_sorted.ptr, ctx.vals.ptr, ctx.vals_sorted.ptr, (int)total, 0, end_bit, stream);
  tile_ranges_kernel<<<div_up(total, 256), 256, 0, stream>>>(total, ctx.keys_sorted, ctx.depth_bits, (uint32_t)ctx.n_tiles, ctx.tile_ranges);
  CUDA_KERNEL_CHECK();
}

}  // namespace b2c
