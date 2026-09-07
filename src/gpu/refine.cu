#include "gpu/refine.h"
#include "gpu/splat_math.cuh"
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <algorithm>

namespace b2c {

namespace {

constexpr float MIN_OPACITY = 1.f / 255.f;
constexpr float MIN_SCALE_FACTOR = 0.1f;

__device__ __forceinline__ float eff_opacity(float raw, float4 ls) {
  float sig = sigmoidf_(raw);
  if (ls.w <= 0.f) return sig;
  float f2 = ls.w * ls.w;
  float3 s2 = make_float3(__expf(2.f * ls.x), __expf(2.f * ls.y), __expf(2.f * ls.z));
  return sig * sqrtf((s2.x * s2.y * s2.z) / ((s2.x + f2) * (s2.y + f2) * (s2.z + f2)));
}
__device__ __forceinline__ float3 eff_scale(float4 ls) {
  float f2 = ls.w * ls.w;
  return make_float3(sqrtf(__expf(2.f * ls.x) + f2), sqrtf(__expf(2.f * ls.y) + f2), sqrtf(__expf(2.f * ls.z) + f2));
}
__device__ __forceinline__ float logit_clamped(float o, float lo, float hi) { o = fminf(fmaxf(o, lo), hi); return __logf(o / (1.f - o)); }

__global__ void min_scale_kernel(int n, const float4* __restrict__ pos, float4* __restrict__ ls, const float* __restrict__ cam_pos, const float* __restrict__ cam_focal, int n_cams) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 p = pos[i];
  float best = INFINITY;
  for (int c = 0; c < n_cams; c++) {
    float dx = p.x - cam_pos[c * 3], dy = p.y - cam_pos[c * 3 + 1], dz = p.z - cam_pos[c * 3 + 2];
    float d = sqrtf(dx * dx + dy * dy + dz * dz) / fmaxf(cam_focal[c], 1e-6f);
    best = fminf(best, d);
  }
  float4 l = ls[i]; l.w = sqrtf(MIN_SCALE_FACTOR) * best; ls[i] = l;
}

__global__ void bake_kernel(int n, float4* __restrict__ pos, float4* __restrict__ ls) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 l = ls[i];
  if (l.w <= 0.f) return;
  float4 p = pos[i];
  float o = eff_opacity(p.w, l);
  p.w = logit_clamped(o, 1e-6f, 1.f - 1e-6f);
  float f2 = l.w * l.w;
  l.x = 0.5f * __logf(__expf(2.f * l.x) + f2); l.y = 0.5f * __logf(__expf(2.f * l.y) + f2); l.z = 0.5f * __logf(__expf(2.f * l.z) + f2); l.w = 0.f;
  pos[i] = p; ls[i] = l;
}

template <int K3>
__global__ void prune_flags_kernel(int n, const float4* __restrict__ pos, const float4* __restrict__ quat, const float4* __restrict__ ls, ShBuf sb,
                                   float3 center, float max_allowed, uint32_t* __restrict__ keep, uint32_t* __restrict__ dead) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 p = pos[i], q = quat[i], l = ls[i];
  bool bad = !(isfinite(p.x) && isfinite(p.y) && isfinite(p.z) && isfinite(p.w) && isfinite(q.x) && isfinite(q.y) && isfinite(q.z) && isfinite(q.w) && isfinite(l.x) && isfinite(l.y) && isfinite(l.z));
  for (int k = 0; k < K3; k++) bad |= !isfinite(sb.get(k, i));
  if (!bad) {
    float o = eff_opacity(p.w, l);
    float3 sc = eff_scale(l);
    bad = o < MIN_OPACITY || sc.x > max_allowed || sc.y > max_allowed || sc.z > max_allowed
          || fabsf(p.x - center.x) > max_allowed || fabsf(p.y - center.y) > max_allowed || fabsf(p.z - center.z) > max_allowed;
  }
  keep[i] = bad ? 0u : 1u; dead[i] = bad ? 1u : 0u;
}

// Gumbel keys for weighted sampling without replacement. key = log(w) + Gumbel; -inf when w <= 0 or excluded.
__global__ void gumbel_keys_kernel(int n, const uint32_t* __restrict__ keep, const uint32_t* __restrict__ excluded, const float4* __restrict__ pos, const float4* __restrict__ ls,
                                   const float* __restrict__ vis_count, const float* __restrict__ refine_norm, int mode, float threshold,
                                   uint32_t seed, float* __restrict__ keys, uint32_t* __restrict__ vals, uint32_t* __restrict__ above) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float w = 0.f;
  bool visible = vis_count[i] > 0.f;
  if (mode == 0) { w = keep[i] && visible ? eff_opacity(pos[i].w, ls[i]) : 0.f; }
  else { bool ab = keep[i] && visible && refine_norm[i] > threshold; if (above) above[i] = ab ? 1u : 0u; w = ab ? refine_norm[i] : 0.f; }
  if (excluded && excluded[i]) w = 0.f;
  float key = -INFINITY;
  if (w > 0.f) {
    float u = fmaxf(hash_uniform(seed, (uint32_t)i, (uint32_t)mode + 7u), 1e-12f);
    key = __logf(w) - __logf(-__logf(u));
  }
  keys[i] = key; vals[i] = (uint32_t)i;
}

__global__ void oversized_flags_kernel(int n, const uint32_t* __restrict__ keep, const uint32_t* __restrict__ excluded, const float* __restrict__ max_screen, const float* __restrict__ vis_count, float thr, uint32_t* __restrict__ flags) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  flags[i] = (keep[i] && !excluded[i] && vis_count[i] > 0.f && max_screen[i] > thr) ? 1u : 0u;
}

__global__ void mark_kernel(int count, const uint32_t* __restrict__ idx, uint32_t* __restrict__ flags) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) flags[idx[i]] = 1u;
}

// Split each selected parent into itself + one child at slot `child_slot`.
template <int K3>
__global__ void split_kernel(int count, const uint32_t* __restrict__ parents, const uint32_t* __restrict__ child_slots,
                             float4* __restrict__ pos, float4* __restrict__ quat, float4* __restrict__ ls, ShBuf sb,
                             float4* __restrict__ m_pos, float4* __restrict__ v_pos, float4* __restrict__ m_q, float4* __restrict__ v_q, float4* __restrict__ m_ls, float4* __restrict__ v_ls,
                             ShBuf msb, float* __restrict__ v_sh, float* __restrict__ refine_norm, float* __restrict__ max_screen, float* __restrict__ vis_count,
                             uint32_t* __restrict__ tile_count, uint32_t* __restrict__ last_step, float split_at_screen_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  uint32_t p = parents[i], c = child_slots[i];
  float4 po = pos[p], q = quat[p], l = ls[p];
  float ms = max_screen[p];
  // Opacity: 1 - (1 - a)^(1/sqrt2), both halves.
  float a = eff_opacity(po.w, l);
  float new_a = 1.f - powf(1.f - a, 0.70710678f);
  float raw = logit_clamped(new_a, 1.f / 255.f, 254.f / 255.f);
  // Scales (effective) and covariance-aware shrink.
  float3 s = eff_scale(l);
  float3 s2 = make_float3(s.x * s.x, s.y * s.y, s.z * s.z);
  float smax = fmaxf(s2.x, fmaxf(s2.y, s2.z));
  float k_max = 0.70710678f;
  if (split_at_screen_size > 0.f && ms > 0.f) k_max = fminf(k_max, split_at_screen_size / ms);
  float3 ratio = make_float3(s2.x / smax, s2.y / smax, s2.z / smax);
  float3 k = make_float3(1.f - ratio.x * (1.f - k_max), 1.f - ratio.y * (1.f - k_max), 1.f - ratio.z * (1.f - k_max));
  float3 off_local = make_float3(sqrtf(fmaxf(0.f, 1.f - k.x * k.x)) * s.x, sqrtf(fmaxf(0.f, 1.f - k.y * k.y)) * s.y, sqrtf(fmaxf(0.f, 1.f - k.z * k.z)) * s.z);
  float qn2 = fmaxf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w, 1e-32f); float inv = rsqrtf(qn2);
  float4 qn = make_float4(q.x * inv, q.y * inv, q.z * inv, q.w * inv);
  Mat3 R = quat_to_mat(qn);
  float3 off = mat3_mul(R, off_local);
  // New log scales: raw log scale of the *effective* scale times k (floor is re-attached after the refine).
  float4 nl = make_float4(__logf(s.x * k.x), __logf(s.y * k.y), __logf(s.z * k.z), 0.f);
  float4 parent = make_float4(po.x - off.x, po.y - off.y, po.z - off.z, raw);
  float4 child = make_float4(po.x + off.x, po.y + off.y, po.z + off.z, raw);
  pos[p] = parent; pos[c] = child;
  quat[c] = qn; quat[p] = qn;
  ls[p] = nl; ls[c] = nl;
  for (int j = 0; j < K3; j++) sb.set(j, c, sb.get(j, p));
  float4 z4 = make_float4(0, 0, 0, 0);
  m_pos[p] = z4; v_pos[p] = z4; m_q[p] = z4; v_q[p] = z4; m_ls[p] = z4; v_ls[p] = z4;
  m_pos[c] = z4; v_pos[c] = z4; m_q[c] = z4; v_q[c] = z4; m_ls[c] = z4; v_ls[c] = z4;
  for (int j = 0; j < K3; j++) { msb.set(j, p, 0.f); msb.set(j, c, 0.f); }
  v_sh[p] = 0.f; v_sh[c] = 0.f;
  refine_norm[c] = 0.f; max_screen[c] = 0.f; vis_count[c] = 0.f; tile_count[c] = 0u; last_step[c] = 0u; last_step[p] = 0u;
}

__global__ void opacity_decay_kernel(int n, float4* __restrict__ pos, float delta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 p = pos[i];
  float o = sigmoidf_(p.w) - delta;
  p.w = logit_clamped(o, 1e-12f, 1.f - 1e-12f);
  pos[i] = p;
}

}  // namespace

void RefineState::init(const Model& m, cudaStream_t stream) { (void)m; (void)stream; counts.reserve(8); h_counts.reserve(8); }

void RefineState::set_cameras(const std::vector<float>& pos, const std::vector<float>& focal) {
  cam_pos.upload(pos); cam_focal.upload(focal); n_cams = (int)focal.size();
  CUDA_CHECK(cudaDeviceSynchronize());
}

void RefineState::update_min_scale(Model& m, cudaStream_t stream) {
  if (n_cams == 0 || m.n == 0) return;
  min_scale_kernel<<<div_up(m.n, 256), 256, 0, stream>>>(m.n, m.pos_op, m.lscale, cam_pos, cam_focal, n_cams);
  CUDA_KERNEL_CHECK();
}

void RefineState::bake_min_scale(Model& m, cudaStream_t stream) {
  if (m.n == 0) return;
  bake_kernel<<<div_up(m.n, 256), 256, 0, stream>>>(m.n, m.pos_op, m.lscale);
  CUDA_KERNEL_CHECK();
}

namespace {
__global__ void axis_kernel(int n, const float4* __restrict__ pos, int axis, float* __restrict__ out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 p = pos[i];
  float v = axis == 0 ? p.x : (axis == 1 ? p.y : p.z);
  out[i] = isfinite(v) ? v : 0.f;
}
}  // namespace

void RefineState::update_bounds(const Model& m, cudaStream_t stream) {
  int n = m.n;
  if (n == 0) return;
  keys.reserve(n); keys_sorted.reserve(n);
  const float percentile = 0.8f;
  size_t lo = (size_t)((1.f - percentile) / 2.f * n), hi = std::min<size_t>(n - 1, (size_t)((1.f + percentile) / 2.f * n));
  h_bounds.reserve(8);
  for (int a = 0; a < 3; a++) {
    axis_kernel<<<div_up(n, 256), 256, 0, stream>>>(n, m.pos_op, a, keys);
    CUDA_KERNEL_CHECK();
    size_t tmp = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, tmp, keys.ptr, keys_sorted.ptr, n, 0, 32, stream);
    cub_tmp.reserve(tmp);
    cub::DeviceRadixSort::SortKeys(cub_tmp.ptr, tmp, keys.ptr, keys_sorted.ptr, n, 0, 32, stream);
    CUDA_CHECK(cudaMemcpyAsync(h_bounds.ptr + a * 2, keys_sorted.ptr + lo, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_bounds.ptr + a * 2 + 1, keys_sorted.ptr + hi, sizeof(float), cudaMemcpyDeviceToHost, stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (int a = 0; a < 3; a++) {
    bounds.min[a] = h_bounds.ptr[a * 2]; bounds.max[a] = h_bounds.ptr[a * 2 + 1];
    bounds.center[a] = 0.5f * (bounds.min[a] + bounds.max[a]); bounds.extent[a] = bounds.max[a] - bounds.min[a];
  }
}

namespace {
uint32_t select_flagged(RefineState& st, const uint32_t* flags, int n, DevBuf<uint32_t>& out, cudaStream_t stream) {
  out.reserve(n);
  thrust::counting_iterator<uint32_t> it(0);
  size_t tmp = 0;
  cub::DeviceSelect::Flagged(nullptr, tmp, it, flags, out.ptr, (uint32_t*)st.counts.ptr, n, stream);
  st.cub_tmp.reserve(tmp);
  cub::DeviceSelect::Flagged(st.cub_tmp.ptr, tmp, it, flags, out.ptr, (uint32_t*)st.counts.ptr, n, stream);
  CUDA_CHECK(cudaMemcpyAsync(st.h_counts.ptr, st.counts.ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return st.h_counts.ptr[0];
}
// Sort Gumbel keys descending, return sorted indices in vals_sorted.
void sort_keys(RefineState& st, int n, cudaStream_t stream) {
  size_t tmp = 0;
  cub::DeviceRadixSort::SortPairsDescending(nullptr, tmp, st.keys.ptr, st.keys_sorted.ptr, st.vals.ptr, st.vals_sorted.ptr, n, 0, 32, stream);
  st.cub_tmp.reserve(tmp);
  cub::DeviceRadixSort::SortPairsDescending(st.cub_tmp.ptr, tmp, st.keys.ptr, st.keys_sorted.ptr, st.vals.ptr, st.vals_sorted.ptr, n, 0, 32, stream);
}
uint32_t count_flags(RefineState& st, const uint32_t* flags, int n, cudaStream_t stream) {
  size_t tmp = 0;
  cub::DeviceReduce::Sum(nullptr, tmp, flags, st.counts.ptr, n, stream);
  st.cub_tmp.reserve(tmp);
  cub::DeviceReduce::Sum(st.cub_tmp.ptr, tmp, flags, st.counts.ptr, n, stream);
  CUDA_CHECK(cudaMemcpyAsync(st.h_counts.ptr, st.counts.ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return st.h_counts.ptr[0];
}
__global__ void append_slots_kernel(int count, uint32_t base, uint32_t* out) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < count) out[i] = base + i; }
__global__ void kill_slots_kernel(int count, const uint32_t* idx, float4* pos) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < count) { float4 p = pos[idx[i]]; p.w = -1e6f; pos[idx[i]] = p; } }
}  // namespace

RefineStats RefineState::run(Model& m, RenderCtx& ctx, const RefineParams& p, cudaStream_t stream) {
  RefineStats stats;
  int n = m.n;
  if (n == 0) return stats;
  refine_count++;
  if (accumulate_min_scale) bake_min_scale(m, stream);
  if (bounds.extent[0] == 0.f) update_bounds(m, stream);
  size_t nn = (size_t)n;
  keep.reserve(nn); dead_flag.reserve(nn); selected.reserve(nn); above.reserve(nn);
  keys.reserve(nn); keys_sorted.reserve(nn); vals.reserve(nn); vals_sorted.reserve(nn);
  sel_parent.reserve(nn * 2 + 16); sel_child.reserve(nn * 2 + 16);
  // 1. Prune flags.
  float max_allowed = bounds.max_extent() * 100.f;
  float3 center = make_float3(bounds.center[0], bounds.center[1], bounds.center[2]);
#define PF(K) prune_flags_kernel<K><<<div_up(n, 256), 256, 0, stream>>>(n, m.pos_op, m.quat, m.lscale, m.sh(), center, max_allowed, keep, dead_flag)
  switch (m.degree) { case 0: PF(3); break; case 1: PF(12); break; case 2: PF(27); break; case 3: PF(48); break; default: PF(75); break; }
#undef PF
  CUDA_KERNEL_CHECK();
  uint32_t n_dead = select_flagged(*this, dead_flag, n, dead_idx, stream);
  stats.pruned = (int)n_dead;
  uint32_t alive = (uint32_t)n - n_dead;
  selected.zero(stream, nn);
  uint32_t n_sel = 0;
  // 2a. Relocation of dead slots: sample n_dead survivors weighted by opacity * visible.
  if (n_dead > 0 && alive > 0) {
    gumbel_keys_kernel<<<div_up(n, 256), 256, 0, stream>>>(n, keep, nullptr, m.pos_op, m.lscale, m.vis_count, m.refine_norm, 0, 0.f, p.seed ^ (p.iter * 2654435761u), keys, vals, nullptr);
    CUDA_KERNEL_CHECK();
    sort_keys(*this, n, stream);
    uint32_t take = std::min<uint32_t>(n_dead, alive);
    CUDA_CHECK(cudaMemcpyAsync(sel_parent.ptr, vals_sorted.ptr, take * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
    mark_kernel<<<div_up(take, 256), 256, 0, stream>>>(take, sel_parent.ptr, selected);
    n_sel = take; stats.relocated = (int)take;
  }
  // 2b. Oversized splits (every refine), capped by headroom.
  if (p.split_at_screen_size > 0.f) {
    oversized_flags_kernel<<<div_up(n, 256), 256, 0, stream>>>(n, keep, selected, m.max_screen, m.vis_count, p.split_at_screen_size, above);
    CUDA_KERNEL_CHECK();
    uint32_t n_over = select_flagged(*this, above, n, tmp_idx, stream);
    uint32_t headroom = alive + n_sel < p.max_splats ? p.max_splats - alive - n_sel : 0;
    n_over = std::min(n_over, headroom);
    if (n_over) {
      CUDA_CHECK(cudaMemcpyAsync(sel_parent.ptr + n_sel, tmp_idx.ptr, n_over * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
      mark_kernel<<<div_up(n_over, 256), 256, 0, stream>>>(n_over, sel_parent.ptr + n_sel, selected);
    }
    n_sel += n_over; stats.split_oversized = (int)n_over;
  }
  // 2c. Gradient-driven growth.
  if (p.growth_allowed) {
    gumbel_keys_kernel<<<div_up(n, 256), 256, 0, stream>>>(n, keep, selected, m.pos_op, m.lscale, m.vis_count, m.refine_norm, 1, p.growth_grad_threshold, p.seed ^ (p.iter * 40503u), keys, vals, above);
    CUDA_KERNEL_CHECK();
    uint32_t n_above = count_flags(*this, above, n, stream);
    int64_t want = (int64_t)std::lround((double)n_above * p.growth_select_fraction) - (int64_t)n_dead;
    if (want < 0) want = 0;
    uint32_t headroom = alive + n_sel < p.max_splats ? p.max_splats - alive - n_sel : 0;
    uint32_t n_grow = (uint32_t)std::min<int64_t>(want, headroom);
    n_grow = std::min(n_grow, n_above);
    if (n_grow) {
      sort_keys(*this, n, stream);
      CUDA_CHECK(cudaMemcpyAsync(sel_parent.ptr + n_sel, vals_sorted.ptr, n_grow * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
    }
    n_sel += n_grow; stats.grown = (int)n_grow;
  }
  // 3. Child slots: reuse dead slots first, then append.
  uint32_t from_dead = std::min(n_dead, n_sel);
  if (n_sel > 0) {
    if (from_dead) CUDA_CHECK(cudaMemcpyAsync(sel_child.ptr, dead_idx.ptr, from_dead * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
    uint32_t appended = n_sel - from_dead;
    int new_n = n + (int)appended;
    if (new_n > m.cap) m.reserve(new_n, stream);
    if (appended) append_slots_kernel<<<div_up(appended, 256), 256, 0, stream>>>(appended, (uint32_t)n, sel_child.ptr + from_dead);
    if (m.cap > (int)ctx.tile_count.count) ctx.setup(ctx.W, ctx.H, m.cap, stream);
    if (appended) CUDA_CHECK(cudaMemsetAsync(ctx.tile_count.ptr + n, 0, appended * sizeof(uint32_t), stream));
#define SK(K) split_kernel<K><<<div_up(n_sel, 256), 256, 0, stream>>>(n_sel, sel_parent, sel_child, m.pos_op, m.quat, m.lscale, m.sh(), m.m_pos_op, m.v_pos_op, m.m_quat, m.v_quat, m.m_lscale, m.v_lscale, m.m_sh(), m.v_sh, m.refine_norm, m.max_screen, m.vis_count, ctx.tile_count, m.last_step, p.split_at_screen_size)
    switch (m.degree) { case 0: SK(3); break; case 1: SK(12); break; case 2: SK(27); break; case 3: SK(48); break; default: SK(75); break; }
#undef SK
    CUDA_KERNEL_CHECK();
    m.n = new_n;
  }
  if (n_dead > from_dead) {
    uint32_t leftover = n_dead - from_dead;
    kill_slots_kernel<<<div_up(leftover, 256), 256, 0, stream>>>(leftover, dead_idx.ptr + from_dead, m.pos_op);
    CUDA_KERNEL_CHECK();
  }
  // 4. Opacity decay on all splats.
  float t = std::min(std::max((float)p.iter / (float)std::max(1u, p.total), 0.f), 1.f);
  float delta = p.opac_decay * (1.f - t);
  opacity_decay_kernel<<<div_up(m.n, 256), 256, 0, stream>>>(m.n, m.pos_op, delta);
  CUDA_KERNEL_CHECK();
  // 5. Bounds and floor.
  update_bounds(m, stream);
  float progress = (float)p.iter / (float)std::max(1u, p.total);
  if (progress < 0.9f) update_min_scale(m, stream);
  // 6. Reset stats.
  m.zero_stats(stream);
  return stats;
}

void bake_min_scale_cpu(SplatCloud& c, const Model& m, cudaStream_t stream) {
  auto ls = m.lscale.download(m.n, stream);
  for (size_t i = 0; i < c.n; i++) {
    float f = ls[i].w;
    if (f <= 0.f) continue;
    float f2 = f * f;
    float s2[3], s2f[3];
    for (int k = 0; k < 3; k++) { s2[k] = std::exp(2.f * c.log_scale[i * 3 + k]); s2f[k] = s2[k] + f2; c.log_scale[i * 3 + k] = 0.5f * std::log(s2f[k]); }
    float coef = std::sqrt((s2[0] * s2[1] * s2[2]) / (s2f[0] * s2f[1] * s2f[2]));
    float o = 1.f / (1.f + std::exp(-c.opacity[i])) * coef;
    o = std::min(std::max(o, 1e-6f), 1.f - 1e-6f);
    c.opacity[i] = std::log(o / (1.f - o));
  }
}

}  // namespace b2c
