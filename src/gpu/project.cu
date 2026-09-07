#include "gpu/render.h"
#include "gpu/splat_math.cuh"

namespace b2c {

namespace {

template <int DEG, int FEAT>  // FEAT: 0 none, 1 normals, 2 buffer
__global__ void project_kernel(int n, const float4* __restrict__ pos_op, const float4* __restrict__ quat, const float4* __restrict__ lscale,
                               const float* __restrict__ sh, const float* __restrict__ feat_in, CamDev cam, bool mip,
                               int tiles_x, int tiles_y,
                               float4* __restrict__ proj0, float4* __restrict__ proj1, float4* __restrict__ proj2, float2* __restrict__ proj3,
                               uint32_t* __restrict__ tile_count, float* __restrict__ max_screen) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 po = pos_op[i], q = quat[i], ls = lscale[i];
  ProjIntermediates pr = project_one(po, q, ls, cam, mip);
  if (!pr.ok) { tile_count[i] = 0; return; }
  float power = __logf(pr.opac * 255.f);
  TileBox bb = tile_bbox(pr.mean2d.x, pr.mean2d.y, pr.ex, pr.ey, tiles_x, tiles_y);
  uint32_t hits = 0;
  for (int ty = bb.min_y; ty < bb.max_y; ty++)
    for (int tx = bb.min_x; tx < bb.max_x; tx++) {
      float rx = tx * (float)TILE_W, ry = ty * (float)TILE_W;
      if (tile_hit(rx, ry, rx + TILE_W, ry + TILE_W, pr.mean2d.x, pr.mean2d.y, pr.conic, power)) hits++;
    }
  tile_count[i] = hits;
  float3 mean = make_float3(po.x, po.y, po.z);
  float3 campos = make_float3(cam.pos[0], cam.pos[1], cam.pos[2]);
  // Colour from SH, view direction camera -> splat.
  float3 v = mean - campos;
  float vl = len3(v); v = v * (1.f / fmaxf(vl, 1e-12f));
  constexpr int K = (DEG + 1) * (DEG + 1);
  float3 col = sh_eval<DEG>(sh + (size_t)i * K * 3, v);
  float cr = col.x + 0.5f, cg = col.y + 0.5f, cb = col.z + 0.5f;
  cr = isfinite(cr) ? fminf(fmaxf(cr, -100.f), 100.f) : 0.f;
  cg = isfinite(cg) ? fminf(fmaxf(cg, -100.f), 100.f) : 0.f;
  cb = isfinite(cb) ? fminf(fmaxf(cb, -100.f), 100.f) : 0.f;
  float3 feat = make_float3(0.f, 0.f, 0.f);
  if constexpr (FEAT == 1) {
    // Pseudo-normal: shortest-scale local axis of R(q), camera-facing, in camera space.
    int axis = (ls.x <= ls.y && ls.x <= ls.z) ? 0 : (ls.y <= ls.z ? 1 : 2);
    float3 a = make_float3(pr.Rq.m[axis], pr.Rq.m[3 + axis], pr.Rq.m[6 + axis]);
    float al = fmaxf(len3(a), 1e-12f);
    float3 u = a * (1.f / al);
    float face = dot3(campos - mean, u) >= 0.f ? 1.f : -1.f;
    Mat3 Rv; for (int k = 0; k < 9; k++) Rv.m[k] = cam.R[k];
    feat = mat3_mul(Rv, u * face);
  } else if constexpr (FEAT == 2) {
    feat = make_float3(feat_in[i * 3], feat_in[i * 3 + 1], feat_in[i * 3 + 2]);
    feat.x = isfinite(feat.x) ? fminf(fmaxf(feat.x, -100.f), 100.f) : 0.f;
    feat.y = isfinite(feat.y) ? fminf(fmaxf(feat.y, -100.f), 100.f) : 0.f;
    feat.z = isfinite(feat.z) ? fminf(fmaxf(feat.z, -100.f), 100.f) : 0.f;
  }
  proj0[i] = make_float4(pr.mean2d.x, pr.mean2d.y, pr.conic.c00, pr.conic.c01);
  proj1[i] = make_float4(pr.conic.c11, pr.opac, pr.mean_c.z, fmaxf(pr.ex / (float)cam.W, pr.ey / (float)cam.H));
  proj2[i] = make_float4(cr, cg, cb, feat.x);
  proj3[i] = make_float2(feat.y, feat.z);
  float ms = fmaxf(pr.ex / (float)cam.W, pr.ey / (float)cam.H);
  if (max_screen) max_screen[i] = fmaxf(max_screen[i], ms);
}

template <int DEG>
void launch_deg(RenderCtx& ctx, const Model& m, const RenderParams& p, const CamDev& cam, cudaStream_t stream) {
  int blocks = div_up(m.n, PROJ_BLOCK);
  float* ms = m.max_screen.ptr;
  switch (p.feat) {
    case FeatureMode::None:
      project_kernel<DEG, 0><<<blocks, PROJ_BLOCK, 0, stream>>>(m.n, m.pos_op, m.quat, m.lscale, m.sh, nullptr, cam, p.mip, ctx.tiles_x, ctx.tiles_y, ctx.proj0, ctx.proj1, ctx.proj2, ctx.proj3, ctx.tile_count, ms); break;
    case FeatureMode::Normals:
      project_kernel<DEG, 1><<<blocks, PROJ_BLOCK, 0, stream>>>(m.n, m.pos_op, m.quat, m.lscale, m.sh, nullptr, cam, p.mip, ctx.tiles_x, ctx.tiles_y, ctx.proj0, ctx.proj1, ctx.proj2, ctx.proj3, ctx.tile_count, ms); break;
    case FeatureMode::Buffer:
      project_kernel<DEG, 2><<<blocks, PROJ_BLOCK, 0, stream>>>(m.n, m.pos_op, m.quat, m.lscale, m.sh, p.feat_buffer, cam, p.mip, ctx.tiles_x, ctx.tiles_y, ctx.proj0, ctx.proj1, ctx.proj2, ctx.proj3, ctx.tile_count, ms); break;
  }
  CUDA_KERNEL_CHECK();
}

}  // namespace

CamDev to_camdev(const CameraGPU& c) {
  CamDev d;
  for (int i = 0; i < 9; i++) d.R[i] = c.R[i];
  for (int i = 0; i < 3; i++) { d.t[i] = c.t[i]; d.pos[i] = c.pos[i]; }
  d.fx = c.fx; d.fy = c.fy; d.cx = c.cx; d.cy = c.cy; d.W = c.W; d.H = c.H;
  d.lim_pos_x = c.lim_pos_x; d.lim_pos_y = c.lim_pos_y; d.lim_neg_x = c.lim_neg_x; d.lim_neg_y = c.lim_neg_y;
  return d;
}

void project_splats(RenderCtx& ctx, const Model& m, const RenderParams& p, cudaStream_t stream) {
  if (m.n == 0) return;
  CamDev cam = to_camdev(p.cam);
  switch (m.degree) {
    case 0: launch_deg<0>(ctx, m, p, cam, stream); break;
    case 1: launch_deg<1>(ctx, m, p, cam, stream); break;
    case 2: launch_deg<2>(ctx, m, p, cam, stream); break;
    case 3: launch_deg<3>(ctx, m, p, cam, stream); break;
    case 4: launch_deg<4>(ctx, m, p, cam, stream); break;
    default: throw std::runtime_error("unsupported SH degree");
  }
}

}  // namespace b2c
