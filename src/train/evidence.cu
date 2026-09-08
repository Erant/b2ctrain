#include "train/evidence.h"
#include "gpu/splat_math.cuh"

namespace b2c {

// Internal accumulator: the exported 7 (w_in, w_all, err, views, dir xyz) plus sum over views of w_in_v^2, from
// which `views` is derived at download as the participation ratio (sum w_in_v)^2 / sum w_in_v^2 -- the effective
// number of views the in-mask mass is spread over. brush counted a view only when it gave the splat >= 1 pixel-weight
// of mass (VIEW_MIN_MASS), which is scale-dependent: a small splat that draws 0.3 of a pixel in every one of 100
// views never counted a single view and was fully distrusted however well it agreed; where the fit was made of such
// splats (the face-priority weight ramp, where the cap and the denoised frames are both attenuated and the trainer
// fits the disagreement with fine splats) the confidence gate culled the pixels. Measured on the stage-2 example
// (2026-09-07, 865k splats, 135 views): 66% of the splats had w_in > 0 and views == 0 and 81% had views < 4; with
// the ratio, 0% and 18%. Culled pixels along the face's rim 10% -> 0%, in the face 1% -> 0%, over the whole subject
// 4.2% -> 3.1% at the anchor view and 3.4% -> 0.5% of the opaque pixels at a view elevated 25 degrees off the orbit
// (the black patches on the specular top). Kept background pixels more than 10 px outside the silhouette: 18 -> 36
// of a 720x1280 frame; the rest of the +12% in kept-outside pixels sits within 10 px of the matte edge. The ratio is
// 1 for a splat seen in one view whatever its mass, N for one seen equally in N views, and a faint tail across many
// views barely moves it (10 in one view + 0.001 in 99 others -> 1.02), which is the floater case the threshold
// existed for.
constexpr int EV_ACC = 8;
static DevBuf<float> g_evidence;   // [n][EV_ACC]
static DevBuf<float> g_view_acc;   // [n][3] per-view (w_in, err, w_all)

namespace {

__device__ __forceinline__ float gt_ch(uint32_t p, int c) { return (float)((p >> (8 * c)) & 0xffu) * (1.f / 255.f); }

// Forward replay per tile accumulating vis * (m, m*res, k) per splat.
template <bool FEAT>
__global__ void __launch_bounds__(RT_PX) evidence_kernel(
    const uint2* __restrict__ tile_ranges, const uint32_t* __restrict__ sorted_vals,
    const float4* __restrict__ proj0, const float4* __restrict__ proj1, const float4* __restrict__ proj2, const float2* __restrict__ proj3,
    const float4* __restrict__ out_rgba, const float4* __restrict__ out_feat, const uint32_t* __restrict__ last_idx,
    const uint32_t* __restrict__ gt, const uint32_t* __restrict__ gtn, const uint8_t* __restrict__ weights,
    int W, int H, int tiles_x, bool masked_has_alpha, float normal_weight, float* __restrict__ acc) {
  __shared__ float4 s0[RT_PX], s1[RT_PX];
  __shared__ uint32_t s_gid[RT_PX];
  __shared__ float s_acc[RT_PX][3];
  const int tile = blockIdx.x;
  const uint2 range = tile_ranges[tile];
  if (range.y <= range.x) return;
  const int tx = tile % tiles_x, ty = tile / tiles_x;
  const int lx = threadIdx.x % RT_W, ly = threadIdx.x / RT_W;
  const int px = tx * RT_W + lx, py = ty * RT_W + ly;
  const bool inside = px < W && py < H;
  const float pcx = px + 0.5f, pcy = py + 0.5f;
  const int lane = threadIdx.x & 31;
  float m = 0.f, mres = 0.f, k = 0.f; uint32_t last = range.x;
  if (inside) {
    int pix = py * W + px;
    uint32_t g = gt[pix];
    float ga = gt_ch(g, 3);
    float wp = weights ? weights[pix] * (1.f / 255.f) : 1.f;
    float4 o = out_rgba[pix];
    float res = (fabsf(o.x - gt_ch(g, 0)) + fabsf(o.y - gt_ch(g, 1)) + fabsf(o.z - gt_ch(g, 2))) * (1.f / 3.f);
    if constexpr (FEAT) {
      if (gtn && normal_weight > 0.f) {
        uint32_t gn = gtn[pix];
        if (((gn >> 24) & 0xffu) > 127u) {
          float gx = (float)(gn & 0xffu) * (2.f / 255.f) - 1.f, gy = 1.f - (float)((gn >> 8) & 0xffu) * (2.f / 255.f), gz = 1.f - (float)((gn >> 16) & 0xffu) * (2.f / 255.f);
          float4 f = out_feat[pix];
          float pl = fmaxf(sqrtf(f.x * f.x + f.y * f.y + f.z * f.z), 1e-6f), gl = fmaxf(sqrtf(gx * gx + gy * gy + gz * gz), 1e-6f);
          float nres = fabsf(f.x - gx) + fabsf(f.y - gy) + fabsf(f.z - gz) + 1.f - (f.x * gx + f.y * gy + f.z * gz) / (pl * gl);
          res += normal_weight * nres;
        }
      }
    }
    m = ga * wp; mres = m * res; k = (masked_has_alpha ? ga : 1.f) * wp;
    last = last_idx[pix];
  }
  float T = 1.f; bool done = !inside;
  for (uint32_t start = range.x; start < range.y; start += RT_PX) {
    if (__syncthreads_count(done) == RT_PX) break;
    uint32_t remaining = min((uint32_t)RT_PX, range.y - start);
    if (threadIdx.x < remaining) {
      uint32_t gid = sorted_vals[start + threadIdx.x];
      s_gid[threadIdx.x] = gid; s0[threadIdx.x] = proj0[gid]; s1[threadIdx.x] = proj1[gid];
      s_acc[threadIdx.x][0] = 0.f; s_acc[threadIdx.x][1] = 0.f; s_acc[threadIdx.x][2] = 0.f;
    }
    __syncthreads();
    for (uint32_t t = 0; t < remaining; t++) {
      float vis = 0.f;
      if (!done && start + t < last) {
        float4 a = s0[t]; float4 c = s1[t];
        float dx = pcx - a.x, dy = pcy - a.y;
        float sigma = 0.5f * (a.z * dx * dx + c.x * dy * dy) + a.w * dx * dy;
        if (sigma >= 0.f && sigma <= c.w) { float alpha = fminf(ALPHA_MAX, c.y * __expf(-sigma)); vis = alpha * T; T *= (1.f - alpha); }
      }
      unsigned ballot = __ballot_sync(0xffffffffu, vis != 0.f);
      if (ballot) {
        float a0 = warp_sum(vis * m), a1 = warp_sum(vis * mres), a2 = warp_sum(vis * k);
        if (lane == 0) { atomicAdd(&s_acc[t][0], a0); atomicAdd(&s_acc[t][1], a1); atomicAdd(&s_acc[t][2], a2); }
      }
    }
    __syncthreads();
    if (threadIdx.x < remaining) {
      float* dst = acc + (size_t)s_gid[threadIdx.x] * 3;
      if (s_acc[threadIdx.x][0] != 0.f) atomicAdd(dst, s_acc[threadIdx.x][0]);
      if (s_acc[threadIdx.x][1] != 0.f) atomicAdd(dst + 1, s_acc[threadIdx.x][1]);
      if (s_acc[threadIdx.x][2] != 0.f) atomicAdd(dst + 2, s_acc[threadIdx.x][2]);
    }
  }
}

__global__ void fold_view_kernel(int n, const float* __restrict__ acc, const float4* __restrict__ pos, float3 cam, float* __restrict__ ev) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float w_in = acc[i * 3], err = acc[i * 3 + 1], w_all = acc[i * 3 + 2];
  float* e = ev + (size_t)i * EV_ACC;
  e[0] += w_in; e[1] += w_all; e[2] += err;
  e[7] += w_in * w_in;
  if (w_in != 0.f) {
    float4 p = pos[i];
    float3 d = make_float3(cam.x - p.x, cam.y - p.y, cam.z - p.z);
    float l = fmaxf(len3(d), 1e-12f);
    e[4] += w_in * d.x / l; e[5] += w_in * d.y / l; e[6] += w_in * d.z / l;
  }
}

}  // namespace

void compute_evidence(RenderCtx& ctx, const Model& m, const std::vector<ViewGPU>& views, const std::vector<Camera>& cams, const Config& cfg, cudaStream_t stream) {
  g_evidence.reserve((size_t)m.n * EV_ACC); g_evidence.zero(stream);
  g_view_acc.reserve((size_t)m.n * 3);
  bool use_normals = cfg.evidence_normal_weight > 0.f;
  for (size_t v = 0; v < views.size(); v++) {
    const ViewGPU& view = views[v];
    if (view.W != ctx.W || view.H != ctx.H) ctx.setup(view.W, view.H, m.cap, stream);
    RenderParams rp; rp.cam = CameraGPU::from(cams[v], view.W, view.H);
    rp.sh_degree = m.degree; rp.bwd_info = true;
    rp.feat = (use_normals && view.normals) ? FeatureMode::Normals : FeatureMode::None;
    render_forward(ctx, m, rp, stream);
    g_view_acc.zero(stream);
    bool masked_alpha = view.masked && view.has_alpha;
    dim3 grid(ctx.n_tiles), block(RT_PX);
    if (rp.feat == FeatureMode::Normals)
      evidence_kernel<true><<<grid, block, 0, stream>>>(ctx.tile_ranges, ctx.vals_sorted, ctx.proj0, ctx.proj1, ctx.proj2, ctx.proj3, ctx.out_rgba, ctx.out_feat, ctx.last_idx, view.rgba, view.normals, view.weights, ctx.W, ctx.H, ctx.tiles_x, masked_alpha, cfg.evidence_normal_weight, g_view_acc);
    else
      evidence_kernel<false><<<grid, block, 0, stream>>>(ctx.tile_ranges, ctx.vals_sorted, ctx.proj0, ctx.proj1, ctx.proj2, ctx.proj3, ctx.out_rgba, ctx.out_feat, ctx.last_idx, view.rgba, nullptr, view.weights, ctx.W, ctx.H, ctx.tiles_x, masked_alpha, 0.f, g_view_acc);
    CUDA_KERNEL_CHECK();
    float3 cp = make_float3(cams[v].pos[0], cams[v].pos[1], cams[v].pos[2]);
    fold_view_kernel<<<div_up(m.n, 256), 256, 0, stream>>>(m.n, g_view_acc, m.pos_op, cp, g_evidence);
    CUDA_KERNEL_CHECK();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

std::vector<float> download_evidence(const Model& m, cudaStream_t stream) {
  std::vector<float> out((size_t)m.n * 7, 0.f);
  if (g_evidence.count < (size_t)m.n * EV_ACC) return out;
  std::vector<float> acc = g_evidence.download((size_t)m.n * EV_ACC, stream);
  for (size_t i = 0; i < (size_t)m.n; i++) {
    const float* a = &acc[i * EV_ACC];
    float* e = &out[i * 7];
    e[0] = a[0]; e[1] = a[1]; e[2] = a[2];
    e[3] = a[7] > 0.f ? (a[0] * a[0]) / a[7] : 0.f;  // effective number of supporting views
    e[4] = a[4]; e[5] = a[5]; e[6] = a[6];
  }
  return out;
}

}  // namespace b2c
