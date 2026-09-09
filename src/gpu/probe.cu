#include "gpu/probe.h"
#include "gpu/splat_math.cuh"

namespace b2c {

namespace {
__global__ void __launch_bounds__(RT_PX) probe_kernel(
    const uint2* __restrict__ tile_ranges, const uint32_t* __restrict__ sorted_vals,
    const float4* __restrict__ proj0, const float4* __restrict__ proj1,
    int W, int H, int tiles_x, float tau, float delta, const float* __restrict__ ref_z, float margin, float4* __restrict__ out) {
  __shared__ float4 s0[RT_PX], s1[RT_PX];
  const int tile = blockIdx.x;
  const int tx = tile % tiles_x, ty = tile / tiles_x;
  const int lx = threadIdx.x % RT_W, ly = threadIdx.x / RT_W;
  const int px = tx * RT_W + lx, py = ty * RT_W + ly;
  const bool inside = px < W && py < H;
  const float pcx = px + 0.5f, pcy = py + 0.5f;
  uint2 range = tile_ranges[tile];
  float T = 1.f, z_first = INFINITY, deep = 0.f, behind = 0.f, zr = INFINITY;
  if (inside && ref_z) zr = ref_z[py * W + px];
  bool done = !inside;
  for (uint32_t start = range.x; start < range.y; start += RT_PX) {
    if (__syncthreads_count(done) == RT_PX) break;
    uint32_t remaining = min((uint32_t)RT_PX, range.y - start);
    if (threadIdx.x < remaining) { uint32_t gid = sorted_vals[start + threadIdx.x]; s0[threadIdx.x] = proj0[gid]; s1[threadIdx.x] = proj1[gid]; }
    __syncthreads();
    for (uint32_t t = 0; t < remaining && !done; t++) {
      float4 a = s0[t]; float4 c = s1[t];
      float dx = pcx - a.x, dy = pcy - a.y;
      float sigma = 0.5f * (a.z * dx * dx + c.x * dy * dy) + a.w * dx * dy;
      if (sigma < 0.f || sigma > c.w) continue;
      float alpha = fminf(ALPHA_MAX, c.y * __expf(-sigma));
      float next_T = T * (1.f - alpha);
      if (next_T <= T_CUTOFF) { done = true; break; }
      float vis = alpha * T;
      float z = c.z;
      if (isfinite(z_first) && z > z_first + delta) deep += vis;
      if (z > zr + margin) behind += vis;
      T = next_T;
      if (!isfinite(z_first) && 1.f - T >= tau) z_first = z;
    }
  }
  if (inside) out[py * W + px] = make_float4(1.f - T, z_first, deep, behind);
}
}  // namespace

void probe_depth(RenderCtx& ctx, float tau, float delta, const float* ref_z, float margin, float4* out, cudaStream_t stream) {
  dim3 grid(ctx.n_tiles), block(RT_PX);
  probe_kernel<<<grid, block, 0, stream>>>(ctx.tile_ranges, ctx.vals_sorted, ctx.proj0, ctx.proj1, ctx.W, ctx.H, ctx.tiles_x, tau, delta, ref_z, margin, out);
  CUDA_KERNEL_CHECK();
}

}  // namespace b2c
