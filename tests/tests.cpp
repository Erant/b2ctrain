// Finite-difference gradient check for the full render + loss + backward path.
#include "gpu/render.h"
#include "gpu/loss.h"
#include "gpu/optim.h"
#include "model.h"
#include "ply.h"
#include "util/log.h"
#include <random>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include "reference.h"

using namespace b2c;

struct Scene {
  int W = 48, H = 40, degree = 2;
  SplatCloud cloud;
  Camera cam;
  std::vector<uint32_t> gt, gtn;
  std::vector<uint8_t> wts;
};

static Scene make_scene(uint64_t seed, int n) {
  Scene s;
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> u(0.f, 1.f), un(-1.f, 1.f);
  s.cloud.resize(n, s.degree);
  int K = s.cloud.K();
  for (int i = 0; i < n; i++) {
    s.cloud.pos[i * 3] = un(rng) * 0.6f; s.cloud.pos[i * 3 + 1] = un(rng) * 0.5f; s.cloud.pos[i * 3 + 2] = 2.0f + un(rng) * 0.5f;
    for (int k = 0; k < 3; k++) s.cloud.log_scale[i * 3 + k] = std::log(0.05f + 0.08f * u(rng));
    float q[4] = {1.f + un(rng) * 0.5f, un(rng) * 0.5f, un(rng) * 0.5f, un(rng) * 0.5f};
    for (int k = 0; k < 4; k++) s.cloud.quat[i * 4 + k] = q[k];
    s.cloud.opacity[i] = un(rng) * 2.f;
    for (int k = 0; k < K * 3; k++) s.cloud.sh[i * K * 3 + k] = (k < 3 ? un(rng) * 1.5f : un(rng) * 0.3f);
  }
  s.cam.width = s.W; s.cam.height = s.H; s.cam.fx = s.cam.fy = 40.f; s.cam.cx = s.W / 2.f; s.cam.cy = s.H / 2.f;
  s.cam.set_w2c_quat(1, 0, 0, 0, 0, 0, 0);
  s.gt.resize(s.W * s.H); s.gtn.resize(s.W * s.H); s.wts.resize(s.W * s.H);
  for (int i = 0; i < s.W * s.H; i++) {
    unsigned r = (unsigned)(u(rng) * 255), g = (unsigned)(u(rng) * 255), b = (unsigned)(u(rng) * 255), a = (unsigned)(u(rng) * 255);
    s.gt[i] = r | (g << 8) | (b << 16) | (a << 24);
    unsigned nr = (unsigned)(u(rng) * 255), ng = (unsigned)(u(rng) * 255), nb = (unsigned)(u(rng) * 255), na = u(rng) > 0.3f ? 255u : 0u;
    s.gtn[i] = nr | (ng << 8) | (nb << 16) | (na << 24);
    s.wts[i] = (uint8_t)(64 + u(rng) * 191);
  }
  return s;
}

struct Harness {
  Model model; RenderCtx ctx; DevBuf<uint32_t> gt, gtn; DevBuf<uint8_t> wts; ViewGPU view; LossParams lp; RenderParams rp; Scene* sc;
  PinnedBuf<float> h; DevBuf<float> grads;
  void setup(Scene& s, bool masked, bool normals) {
    sc = &s;
    model.upload(s.cloud);
    ctx.setup(s.W, s.H, model.cap, 0);
    gt.upload(s.gt); gtn.upload(s.gtn); wts.upload(s.wts);
    view.rgba = gt; view.normals = normals ? gtn.ptr : nullptr; view.weights = wts; view.W = s.W; view.H = s.H; view.has_alpha = true; view.masked = masked; view.alpha_coverage = 0.5f; view.normal_count = 100.f;
    lp.l1_w = 0.8f; lp.ssim_w = -0.2f; lp.bg[0] = 0.2f; lp.bg[1] = 0.1f; lp.bg[2] = 0.3f;
    lp.composite = !masked; lp.mask = masked; lp.alpha_lane = !masked; lp.match_alpha_weight = 0.3f; lp.scale = masked ? 1.7f : 1.f;
    lp.normal_scale = normals ? 0.05f / 100.f : 0.f;
    rp.cam = CameraGPU::from(s.cam, s.W, s.H); for (int k = 0; k < 3; k++) rp.bg[k] = lp.bg[k];
    rp.sh_degree = s.degree; rp.feat = normals ? FeatureMode::Normals : FeatureMode::None; rp.bwd_info = true;
    h.reserve(16);
  }
  double loss() {
    render_forward(ctx, model, rp, 0);
    ctx.loss_accum.zero(0, 4);
    photometric_loss(ctx, view, lp, 0);
    if (lp.normal_scale > 0) normal_loss(ctx, view, lp, 0);
    CUDA_CHECK(cudaMemcpy(h.ptr, ctx.loss_accum.ptr, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    return (double)h.ptr[0] + (double)h.ptr[1];
  }
  bool tc = false;
  std::vector<float> analytic() {
    int K = model.K(); size_t per = 11 + K * 3;
    grads.reserve((size_t)model.n * per); grads.zero();
    lp.grad_scale = tc ? 3.f * sc->W * sc->H : 1.f;
    loss();
    ctx.v_feat.zero();
    if (lp.normal_scale > 0) normal_loss(ctx, view, lp, 0);
    if (tc) rasterize_backward_tc(ctx, model, rp, lp.grad_scale, 0); else rasterize_backward(ctx, model, rp, 0);
    lp.grad_scale = 1.f;
    OptimParams op; op.cam = rp.cam; op.active_sh_degree = model.degree; op.grad_out = grads.ptr;
    optimizer_step(ctx, model, op, 0);
    return grads.download();
  }
};

static float g_eps = 1e-4f;
int main(int argc, char** argv) {
  if (argc > 1) g_eps = (float)atof(argv[1]);
  int fails = 0, total = 0, skipped = 0;
  for (int cfg = 0; cfg < 6; cfg++) {
    bool masked = cfg % 3 == 1, normals = cfg % 3 == 2, tc = cfg >= 3;
    Scene s = make_scene(1234 + cfg % 3, 24);
    Harness hs; hs.setup(s, masked, normals); hs.tc = tc;
    std::vector<float> an = hs.analytic();
    RefParams rp; rp.W = s.W; rp.H = s.H; rp.cam = s.cam; for (int k = 0; k < 3; k++) rp.bg[k] = hs.lp.bg[k];
    rp.composite = hs.lp.composite; rp.mask = hs.lp.mask; rp.alpha_lane = hs.lp.alpha_lane; rp.normals = normals;
    rp.l1_w = hs.lp.l1_w; rp.ssim_w = hs.lp.ssim_w; rp.match_alpha_weight = hs.lp.match_alpha_weight; rp.scale = hs.lp.scale; rp.normal_scale = hs.lp.normal_scale;
    rp.gt = &s.gt; rp.gtn = &s.gtn; rp.wts = &s.wts;
    double gpu_loss = hs.loss(), ref_loss = reference_loss(s.cloud, rp);
    printf("cfg %d: gpu loss %.7f reference loss %.7f (rel diff %.2e)\n", cfg, gpu_loss, ref_loss, std::abs(gpu_loss - ref_loss) / std::abs(ref_loss));
    int K = s.cloud.K(); size_t per = 11 + K * 3;
    const char* names[] = {"pos.x", "pos.y", "pos.z", "opac", "q.w", "q.x", "q.y", "q.z", "ls.x", "ls.y", "ls.z"};
    double max_rel = 0; int checked = 0;
    for (int i = 0; i < s.cloud.n; i += 2) {
      for (size_t p = 0; p < per; p++) {
        if (p >= 11 && (p - 11) % 5 != 0) continue;  // sample SH lanes
        float* ref = nullptr;
        if (p < 3) ref = &s.cloud.pos[i * 3 + p];
        else if (p == 3) ref = &s.cloud.opacity[i];
        else if (p < 8) ref = &s.cloud.quat[i * 4 + (p - 4)];
        else if (p < 11) ref = &s.cloud.log_scale[i * 3 + (p - 8)];
        else ref = &s.cloud.sh[i * K * 3 + (p - 11)];
        float orig = *ref; float eps = g_eps;
        auto fd_at = [&](float e) { *ref = orig + e; double lp = reference_loss(s.cloud, rp); *ref = orig - e; double lm = reference_loss(s.cloud, rp); *ref = orig; return (lp - lm) / (2.0 * e); };
        double fd = fd_at(eps), fd2 = fd_at(4.f * eps);
        // A discontinuity inside the stencil makes the two estimates disagree; skip those points.
        if (std::abs(fd - fd2) > 0.1 * std::max(std::abs(fd), std::abs(fd2)) + 1e-6) { skipped++; continue; }
        double a = an[i * per + p];
        double denom = std::max(std::abs(fd), std::abs(a));
        double rel = denom > 1e-7 ? std::abs(fd - a) / denom : 0.0;
        total++; checked++;
        bool bad = rel > 0.02 && std::abs(fd - a) > 1e-5;
        if (bad) { fails++; printf("  MISMATCH cfg%d splat %d %s: analytic %.6g fd %.6g rel %.3f\n", cfg, i, p < 11 ? names[p] : "sh", a, fd, rel); }
        max_rel = std::max(max_rel, rel);
      }
    }
    printf("cfg %d (%s%s%s): checked %d params, max rel err %.4f\n", cfg, masked ? "masked" : "transparent", normals ? "+normals" : "", tc ? " tc" : " warp", checked, max_rel);
  }
  printf("%d / %d mismatches (%d non-smooth points skipped)\n", fails, total, skipped);
  return fails == 0 ? 0 : 1;
}
