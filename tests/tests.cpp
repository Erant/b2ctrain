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
#include "gpu/align.h"
#include "gpu/deform.h"
#include "train/evidence.h"
#include "cli.h"
#include <cstdio>

using namespace b2c;

struct Scene {
  int W = 48, H = 40, degree = 2;
  SplatCloud cloud;
  Camera cam;
  std::vector<uint32_t> gt, gtn;
  std::vector<uint8_t> wts;
  std::vector<float> hollow_z;  // per-pixel reference surface depth for the hollow loss (inf = none)
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
  s.hollow_z.resize(s.W * s.H);
  for (int y = 0; y < s.H; y++) for (int x = 0; x < s.W; x++) s.hollow_z[y * s.W + x] = (x < 6 && y < 6) ? INFINITY : 1.9f + 0.25f * std::sin(x * 0.4f) + 0.2f * std::cos(y * 0.3f);
  return s;
}

struct Harness {
  Model model; RenderCtx ctx; DevBuf<uint32_t> gt, gtn; DevBuf<uint8_t> wts; DevBuf<float> hz; ViewGPU view; LossParams lp; RenderParams rp; Scene* sc;
  PinnedBuf<float> h; DevBuf<float> grads;
  bool hollow = false;
  void setup(Scene& s, bool masked, bool normals, bool with_hollow = false) {
    sc = &s; hollow = with_hollow;
    model.upload(s.cloud);
    ctx.setup(s.W, s.H, model.cap, 0);
    gt.upload(s.gt); gtn.upload(s.gtn); wts.upload(s.wts);
    view.rgba = gt; view.normals = normals ? gtn.ptr : nullptr; view.weights = wts; view.W = s.W; view.H = s.H; view.has_alpha = true; view.masked = masked; view.alpha_coverage = 0.5f; view.normal_count = 100.f;
    lp.l1_w = 0.8f; lp.ssim_w = -0.2f; lp.bg[0] = 0.2f; lp.bg[1] = 0.1f; lp.bg[2] = 0.3f;
    lp.composite = !masked; lp.mask = masked; lp.alpha_lane = !masked; lp.match_alpha_weight = 0.3f; lp.scale = masked ? 1.7f : 1.f;
    lp.normal_scale = normals ? 0.05f / 100.f : 0.f;
    rp.cam = CameraGPU::from(s.cam, s.W, s.H); for (int k = 0; k < 3; k++) rp.bg[k] = lp.bg[k];
    rp.sh_degree = s.degree; rp.feat = normals ? FeatureMode::Normals : FeatureMode::None; rp.bwd_info = true;
    if (hollow) { hz.upload(s.hollow_z); rp.hollow_z = hz; rp.hollow_margin = 0.05f; rp.hollow_lam = 0.7f / (float)(s.W * s.H); }
    h.reserve(16);
  }
  double loss() {
    render_forward(ctx, model, rp, 0);
    ctx.loss_accum.zero(0, 4);
    photometric_loss(ctx, view, lp, 0);
    if (lp.normal_scale > 0) normal_loss(ctx, view, lp, 0);
    if (hollow) hollow_loss(ctx, rp.hollow_lam, 0);
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

// Flow + warp against a synthetic shift (the properties pipeline/align.py's tests check): a frame that agrees with its
// render is untouched, and a frame whose texture sits two pixels off its render comes back sitting on it.
static int test_align() {
  const int W = 96, H = 96;
  std::mt19937 rng(7); std::uniform_real_distribution<float> u(0.f, 255.f);
  std::vector<float> noise(W * H), tex(W * H);
  for (auto& v : noise) v = u(rng);
  for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {  // 5x5 box blur: enough structure to match on, not pure noise
    float s = 0.f; int c = 0;
    for (int dy = -2; dy <= 2; dy++) for (int dx = -2; dx <= 2; dx++) { int xx = std::min(std::max(x + dx, 0), W - 1), yy = std::min(std::max(y + dy, 0), H - 1); s += noise[yy * W + xx]; c++; }
    tex[y * W + x] = s / c;
  }
  auto pixel = [&](int x, int y) {  // r = tex, g = tex rolled 3 rows, b = tex rolled 3 columns (as the Python fixture)
    unsigned r = (unsigned)tex[y * W + x], g = (unsigned)tex[((y + H - 3) % H) * W + x], b = (unsigned)tex[y * W + (x + W - 3) % W];
    return r | (g << 8) | (b << 16) | (255u << 24);
  };
  std::vector<uint32_t> frame(W * H);
  for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) frame[y * W + x] = pixel(x, y);
  auto as_render = [&](int shift) {  // np.roll(frame, shift, axis=1) as float4 on grey (alpha is 255 everywhere)
    std::vector<float4> r(W * H);
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) { uint32_t p = frame[y * W + (x - shift + W) % W]; r[y * W + x] = make_float4((p & 255) / 255.f, ((p >> 8) & 255) / 255.f, ((p >> 16) & 255) / 255.f, 1.f); }
    return r;
  };
  auto diff = [&](const std::vector<uint32_t>& a, const std::vector<float4>& b) {  // mean |a - b| over the interior, per channel
    double s = 0; int c = 0;
    for (int y = 8; y < H - 8; y++) for (int x = 8; x < W - 8; x++) { uint32_t p = a[y * W + x]; float4 q = b[y * W + x];
      s += std::abs((p & 255) - q.x * 255.f) + std::abs(((p >> 8) & 255) - q.y * 255.f) + std::abs(((p >> 16) & 255) - q.z * 255.f); c += 3; }
    return s / c;
  };
  DevBuf<uint32_t> d_frame, d_out; d_frame.upload(frame); d_out.reserve(W * H);
  DevBuf<float4> d_render;
  AlignScratch scratch;
  int fails = 0;
  {
    auto render = as_render(0); d_render.upload(render);
    AlignStats st = align_view_gpu(scratch, d_frame, d_render, W, H, 6.f, 6.f, d_out, 0);
    auto out = d_out.download(W * H);
    bool same = out == frame;
    printf("align identity: mean %.4f px, warped %s the frame\n", st.mean, same ? "==" : "!=");
    if (!same || st.mean > 1e-3f) fails++;
  }
  {
    auto render = as_render(2); d_render.upload(render);
    AlignStats st = align_view_gpu(scratch, d_frame, d_render, W, H, 6.f, 6.f, d_out, 0);
    auto out = d_out.download(W * H);
    double before = diff(frame, render), after = diff(out, render);
    printf("align shift 2: measured %.3f px mean (p90 %.3f), residual %.2f -> %.2f\n", st.mean, st.p90, before, after);
    if (std::abs(st.mean - 2.f) > 0.6f || after > before / 2) fails++;
  }
  {
    auto render = as_render(2); d_render.upload(render);
    const int gw = align_warp_dim(W), gh = align_warp_dim(H);
    DevBuf<float2> d_grid; d_grid.reserve((size_t)gw * gh);
    AlignStats st = align_view_gpu(scratch, d_frame, d_render, W, H, 6.f, 6.f, nullptr, 0, d_grid);  // render-side: grid only
    auto grid = d_grid.download((size_t)gw * gh);
    double sx = 0, sy = 0; int c = 0;
    for (int y = 2; y < gh - 2; y++) for (int x = 2; x < gw - 2; x++) { sx += grid[y * gw + x].x; sy += grid[y * gw + x].y; c++; }
    sx /= c; sy /= c;
    printf("align grid: %dx%d cells, interior mean (%.3f, %.3f) px for a 2 px shift (measured %.3f)\n", gw, gh, sx, sy, st.mean);
    if (std::abs(sx - 2.0) > 0.6 || std::abs(sy) > 0.3) fails++;
  }
  {
    auto render = as_render(5); d_render.upload(render);
    AlignStats st = align_view_gpu(scratch, d_frame, d_render, W, H, 6.f, 1.f, d_out, 0);  // cap 1 px: measured 5, applied <= 1
    auto out = d_out.download(W * H);
    auto one = as_render(1);
    double to_render = diff(out, render), to_one = diff(out, one);
    printf("align cap: measured %.3f px, warped is %.2f from the render and %.2f from a 1 px shift\n", st.mean, to_render, to_one);
    if (st.mean < 3.f || to_one > to_render) fails++;
  }
  return fails;
}

// Articulated deformation: a chain root -> j1 -> j2 with j1 active; a 90-degree rotation about j1's pivot moves a
// point bound to j2 as expected; a gradient on that point produces the torque r x g about j1, and Adam's first step
// on the rotation follows it (Adam: lr * sign).
static int test_deform() {
  const char* path = "/tmp/b2c_test_rig.bin";
  {
    FILE* f = fopen(path, "wb");
    fwrite("B2CRIG2\0", 1, 8, f);
    int32_t hdr[6] = {2, 3, 1, 4, 64, 1}; fwrite(hdr, sizeof(hdr), 1, f);
    float verts[6] = {0.f, 2.5f, 0.f, 0.f, 0.5f, 0.f}; fwrite(verts, sizeof(verts), 1, f);
    int32_t vj[8] = {2, 0, 0, 0, 0, 0, 0, 0}; fwrite(vj, sizeof(vj), 1, f);
    float vw[8] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f}; fwrite(vw, sizeof(vw), 1, f);
    char name[64] = "view0.png"; fwrite(name, 64, 1, f);
    int32_t parents[3] = {-1, 0, 1}; fwrite(parents, sizeof(parents), 1, f);
    float jpos[9] = {0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 2.f, 0.f}; fwrite(jpos, sizeof(jpos), 1, f);
    int32_t active[1] = {1}; fwrite(active, sizeof(active), 1, f);
    fclose(f);
  }
  BodyRig rig;
  if (!rig.load(path)) { printf("deform: cannot load the test rig\n"); return 1; }
  SplatCloud c; c.resize(2, 0);
  float pos[6] = {0.f, 2.5f, 0.f, 0.f, 0.5f, 0.f};
  for (int i = 0; i < 2; i++) { for (int k = 0; k < 3; k++) { c.pos[i * 3 + k] = pos[i * 3 + k]; c.log_scale[i * 3 + k] = -3.f; } c.quat[i * 4] = 1.f; c.opacity[i] = 0.5f; }
  Model m; m.upload(c);
  rig.bind(m, 0);
  const float half_pi = 1.57079633f;
  std::vector<float3> om(3, make_float3(0.f, 0.f, 0.f)); om[1] = make_float3(0.f, 0.f, half_pi);
  rig.omega.upload(om);
  rig.pose(0, m, 0);
  auto pv = rig.pos_view.download(2);
  int fails = 0;
  // (0, 2.5, 0) about pivot (0, 1, 0) by +90 deg around z: r = (0, 1.5, 0) -> (-1.5, 0, 0) -> (-1.5, 1, 0); the root-bound point stays
  bool ok0 = std::abs(pv[0].x + 1.5f) < 1e-4f && std::abs(pv[0].y - 1.f) < 1e-4f && std::abs(pv[0].z) < 1e-4f && std::abs(pv[0].w - 0.5f) < 1e-6f;
  bool ok1 = std::abs(pv[1].x) < 1e-6f && std::abs(pv[1].y - 0.5f) < 1e-6f;
  printf("deform pose: bound point -> (%.3f, %.3f, %.3f) %s, root point %s\n", pv[0].x, pv[0].y, pv[0].z, ok0 ? "ok" : "WRONG", ok1 ? "unmoved" : "MOVED");
  if (!ok0 || !ok1) fails++;
  // torque, at a 0.3 rad rotation (the update clamps rotations to 0.6 rad): g = (0, 1, 0) on the bound point gives
  // r x g = (0, 0, r.x) with r.x = -1.5 sin 0.3 < 0, so Adam's first step (lr * sign) raises omega_z by lr
  om[1] = make_float3(0.f, 0.f, 0.3f); rig.omega.upload(om);
  rig.pose(0, m, 0);
  std::vector<float3> g = {make_float3(0.f, 1.f, 0.f), make_float3(0.f, 0.f, 0.f)};
  rig.g_pos.upload(g);
  rig.update(0, m, 1e-3f, 0.f, 0.f, false, 0);
  auto om2 = rig.download_omega(0);
  bool ok2 = std::abs(om2[1].z - (0.3f + 1e-3f)) < 1e-5f && std::abs(om2[1].x) < 1e-6f && std::abs(om2[1].y) < 1e-6f && om2[0].z == 0.f && om2[2].z == 0.f;
  printf("deform torque: omega_z %.6f (expected %.6f), inactive joints untouched %s\n", om2[1].z, 0.3f + 1e-3f, ok2 ? "yes" : "NO");
  if (!ok2) fails++;
  // v3: the same rig with a per-view displacement of the bound vertex, (0, 0.5, 0) for vertex 0. It is added BEFORE the
  // blend: the bound point becomes (0, 3, 0), r = (0, 2, 0) about the pivot -> (-2, 1, 0) at 90 deg.
  {
    const char* path3 = "/tmp/b2c_test_rig3.bin";
    FILE* f = fopen(path3, "wb");
    fwrite("B2CRIG3\0", 1, 8, f);
    int32_t hdr[6] = {2, 3, 1, 4, 64, 1}; fwrite(hdr, sizeof(hdr), 1, f);
    float verts[6] = {0.f, 2.5f, 0.f, 0.f, 0.5f, 0.f}; fwrite(verts, sizeof(verts), 1, f);
    int32_t vj[8] = {2, 0, 0, 0, 0, 0, 0, 0}; fwrite(vj, sizeof(vj), 1, f);
    float vw[8] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f}; fwrite(vw, sizeof(vw), 1, f);
    char name[64] = "view0.png"; fwrite(name, 64, 1, f);
    int32_t parents[3] = {-1, 0, 1}; fwrite(parents, sizeof(parents), 1, f);
    float jpos[9] = {0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 2.f, 0.f}; fwrite(jpos, sizeof(jpos), 1, f);
    int32_t active[1] = {1}; fwrite(active, sizeof(active), 1, f);
    float delta[6] = {0.f, 0.5f, 0.f, 0.f, 0.f, 0.f}; fwrite(delta, sizeof(delta), 1, f);
    fclose(f);
    BodyRig rig3;
    if (!rig3.load(path3) || !rig3.has_delta) { printf("deform v3: cannot load the test rig\n"); return fails + 1; }
    rig3.bind(m, 0);
    std::vector<float3> om3(3, make_float3(0.f, 0.f, 0.f)); om3[1] = make_float3(0.f, 0.f, half_pi);
    rig3.omega.upload(om3);
    rig3.pose(0, m, 0);
    auto pv3 = rig3.pos_view.download(2);
    bool ok3 = std::abs(pv3[0].x + 2.f) < 1e-4f && std::abs(pv3[0].y - 1.f) < 1e-4f && std::abs(pv3[0].z) < 1e-4f && std::abs(pv3[1].y - 0.5f) < 1e-6f;
    printf("deform v3 delta: bound point -> (%.3f, %.3f, %.3f) %s\n", pv3[0].x, pv3[0].y, pv3[0].z, ok3 ? "ok" : "WRONG");
    if (!ok3) fails++;
  }
  return fails;
}

// The label vote: a frame whose left half is class 3 and right half class 13 must label every splat
// by the side its centre projects to, at full confidence away from the seam; the seg_label/seg_conf
// columns must survive a ply round trip.
static int test_labels() {
  int fails = 0;
  Scene s = make_scene(11, 60);
  Model model; model.upload(s.cloud);
  RenderCtx ctx; ctx.setup(s.W, s.H, model.cap, 0);
  DevBuf<uint32_t> gt; gt.upload(s.gt);
  std::vector<uint8_t> lab(s.W * s.H);
  for (int y = 0; y < s.H; y++) for (int x = 0; x < s.W; x++) lab[y * s.W + x] = x < s.W / 2 ? 3 : 13;
  DevBuf<uint8_t> dlab; dlab.upload(lab);
  ViewGPU view; view.rgba = gt; view.labels = dlab; view.W = s.W; view.H = s.H; view.has_alpha = true; view.masked = false;
  std::vector<ViewGPU> views{view}; std::vector<Camera> cams{s.cam};
  Config cfg; cfg.export_labels = true;
  compute_evidence(ctx, model, views, cams, cfg, 0);
  std::vector<float> lv = download_labels(model, 0);
  int checked = 0, voted = 0;
  for (int i = 0; i < model.n; i++) {
    float x = s.cloud.pos[i * 3], z = s.cloud.pos[i * 3 + 2];
    float px = s.cam.fx * x / z + s.cam.cx;
    if (lv[i * 2 + 1] > 0.f) voted++;
    if (std::abs(px - s.W / 2.f) < 6.f || px < 2 || px > s.W - 2) continue;  // straddles the seam or the frame edge
    checked++;
    int want = px < s.W / 2.f ? 3 : 13;
    bool ok = (int)lv[i * 2] == want && lv[i * 2 + 1] > 0.9f;
    if (!ok) { fails++; printf("labels: splat %d at px %.1f voted %d (conf %.3f), expected %d\n", i, px, (int)lv[i * 2], lv[i * 2 + 1], want); }
  }
  printf("label vote: %d/%d splats voted, %d away from the seam checked, %d wrong\n", voted, model.n, checked, fails);
  SplatCloud c = model.download(0); c.has_labels = true; c.labels = lv;
  const char* path = "/tmp/b2c_test_labels.ply";
  write_ply(path, c, {"SH degree: 2"});
  SplatCloud back = read_ply(path);
  bool rt = back.has_labels && back.n == c.n;
  for (size_t i = 0; rt && i < c.n * 2; i++) if (back.labels[i] != c.labels[i]) rt = false;
  printf("label ply round trip: %s\n", rt ? "ok" : "MISMATCH");
  if (!rt) fails++;
  std::remove(path);
  return fails;
}

int main(int argc, char** argv) {
  int align_fails = test_align();
  int deform_fails = test_deform();
  if (deform_fails) printf("deform: %d failure(s)\n", deform_fails);
  if (argc > 1) g_eps = (float)atof(argv[1]);
  int fails = 0, total = 0, skipped = 0;
  for (int cfg = 0; cfg < 9; cfg++) {
    bool masked = cfg % 3 == 1, normals = cfg % 3 == 2, tc = cfg >= 3, hollow = cfg >= 6;
    Scene s = make_scene(1234 + cfg % 3, 24);
    Harness hs; hs.setup(s, masked, normals, hollow); hs.tc = tc;
    std::vector<float> an = hs.analytic();
    RefParams rp; rp.W = s.W; rp.H = s.H; rp.cam = s.cam; for (int k = 0; k < 3; k++) rp.bg[k] = hs.lp.bg[k];
    rp.composite = hs.lp.composite; rp.mask = hs.lp.mask; rp.alpha_lane = hs.lp.alpha_lane; rp.normals = normals;
    rp.l1_w = hs.lp.l1_w; rp.ssim_w = hs.lp.ssim_w; rp.match_alpha_weight = hs.lp.match_alpha_weight; rp.scale = hs.lp.scale; rp.normal_scale = hs.lp.normal_scale;
    rp.gt = &s.gt; rp.gtn = &s.gtn; rp.wts = &s.wts;
    if (hollow) { rp.hollow_z = &s.hollow_z; rp.hollow_lam = hs.rp.hollow_lam; rp.hollow_margin = hs.rp.hollow_margin; }
    double gpu_loss = hs.loss(), ref_loss = reference_loss(s.cloud, rp);
    printf("cfg %d: gpu loss %.7f reference loss %.7f (rel diff %.2e)\n", cfg, gpu_loss, ref_loss, std::abs(gpu_loss - ref_loss) / std::abs(ref_loss));
    int K = s.cloud.K(); size_t per = 11 + K * 3;
    const char* names[] = {"pos.x", "pos.y", "pos.z", "opac", "q.w", "q.x", "q.y", "q.z", "ls.x", "ls.y", "ls.z"};
    double max_rel = 0; int checked = 0;
    for (int i = 0; i < s.cloud.n; i += 2) {
      for (size_t p = 0; p < per; p++) {
        if (p >= 11 && (p - 11) % 5 != 0) continue;  // sample SH lanes
        // The hollow loss treats a fragment's depth as a constant: its gradient runs through the compositing weights
        // only (pulling a penalised splat towards the camera would drag the far side of the body forward through it),
        // so the finite difference along the means, which moves the depth, is not what the kernel computes.
        if (hollow && p < 3) continue;
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
    printf("cfg %d (%s%s%s%s): checked %d params, max rel err %.4f\n", cfg, masked ? "masked" : "transparent", normals ? "+normals" : "", tc ? " tc" : " warp", hollow ? " +hollow" : "", checked, max_rel);
  }
  printf("%d / %d mismatches (%d non-smooth points skipped)\n", fails, total, skipped);
  fails += test_labels();
  if (align_fails) printf("%d alignment test failure(s)\n", align_fails);
  return (fails == 0 && align_fails == 0 && deform_fails == 0) ? 0 : 1;
}
