// b2ctrain fit-cage: fit per-frame cage displacements to a generated video, the splat frozen.
//
// The subject's splat is fixed (appearance and canonical geometry); a cage file (gpu/cage.h) poses it per frame. The
// video's frames (a COLMAP dataset whose image names are the cage's frame names, with alpha from masks/ and Sapiens2
// labels from labels/) are what the posed splat should look like. Several views of a frame fit it jointly: an image
// named "<frame>@<anything>" belongs to cage frame <frame>. For each frame the fit learns a displacement of
// every cage vertex so the render matches the frame:
//   * photometric + alpha: the trainer's L1/SSIM and alpha lane (low weight by default: the video drifts in colour);
//   * labels: the splats render a 3-channel one-hot of their seg_label's GROUP (--groups) and are pulled towards the
//     frame's labels of the same groups; this is the term that says where the jacket or the hair is, whatever colour
//     the video gave it.
// The splat's positional gradient reaches its triangle's vertices by barycentric weight (gpu/cage.h). Adam per
// (frame, vertex) with one second moment per vertex shared by all frames (a vertex hidden in a frame takes a small
// step there), then three proximal pulls, each a blend towards a target so their strength does not depend on the
// scale of the data gradient: towards the mean of the vertex's mesh neighbours (--lap), towards the mean of the
// neighbouring frames (--temporal), and towards zero (--zero).
//
// Outputs in --output: delta.f32 [frames][verts][3], vis.f32 [frames][verts] (barycentric sum of the bound splats
// that received gradient, i.e. were seen), fit.json (per-frame losses before/after, the frame names).
#include "train/fit_cage.h"
#include "gpu/cage.h"
#include "gpu/render.h"
#include "gpu/loss.h"
#include "gpu/optim.h"
#include "train/gpu_views.h"
#include "dataset/views.h"
#include "cli.h"
#include "ply.h"
#include "util/log.h"
#include "json.hpp"
#include <filesystem>
#include <fstream>
#include <random>
#include <cstring>
#include <cmath>

namespace b2c {
namespace fs = std::filesystem;
namespace {

struct FitArgs {
  std::string splat, cage, dataset, output;
  uint32_t iters = 6000;
  float lr = 3e-4f;                 // scene units per step (Adam, one shared second moment)
  float lap = 0.02f, temporal = 0.1f, zero = 0.002f, max_disp = 0.2f, step_clip = 4.f;
  std::vector<int> freeze_layers;
  std::string alt_binding;          // dual binding file (gpu/cage.h); with fit_binding its weights are learned
  bool fit_binding = false, no_delta = false;
  float binding_lr = 0.05f;         // Adam step on the weights' logits
  std::string appearance_mask;      // uint8 per splat: these splats' opacity and colour are learned (the rest stay frozen)
  float app_lr_dc = 2e-3f, app_lr_opac = 0.012f, app_lr_sh = 0.f;
  bool fit_splats = false;          // with the mask: also the splats' means, rotations and scales (canonical frame)
  float sp_lr_mean = 1e-4f, sp_lr_rot = 1e-3f, sp_lr_scale = 5e-3f;
  int grad_smooth = 50;             // Jacobi passes smoothing the per-vertex gradient over the mesh before the step
  float photo_weight = 0.3f, alpha_weight = 1.0f, label_weight = 1.0f, ssim_weight = 0.2f;
  float min_conf = 0.5f;
  std::vector<std::vector<int>> groups;   // three label groups (b2crig chooses them: b2crig.b2ctrain.FIT_GROUPS)
  uint32_t max_resolution = 1920;
  bool res_schedule = true;
  int device = 0; uint64_t seed = 42;
};

const char* HELP =
"Fit per-frame cage displacements so the frozen splat, posed by the cage, matches a video\n\n"
"Usage: b2ctrain fit-cage --splat PLY --cage CAGE --dataset DIR --output DIR [OPTIONS]\n\n"
"Options:\n"
"      --splat PLY          The subject's trained splat (with seg_label)\n"
"      --cage CAGE          Cage file (gpu/cage.h); its frame names are the dataset's image names\n                           (an image \"<frame>@<view>\" is another view of <frame>)\n"
"      --dataset DIR        COLMAP text dataset of the video (images with alpha or masks/, labels/ sidecar)\n"
"      --output DIR         Where delta.f32, vis.f32 and fit.json go\n"
"      --iters N            Steps (one frame each) [default: 6000]\n"
"      --lr M               Adam step in scene units [default: 0.0003]\n"
"      --lap B              Per-step blend of the displacement towards its mesh-neighbour mean [default: 0.02]\n"
"      --grad-smooth N      Jacobi passes (blend 0.5 towards the neighbour mean) over the per-vertex gradient before each step:\n"
"                           spreads the evidence of the seen splats over the surface instead of damping the displacement [default: 50]\n"
"      --temporal B         Per-step blend towards the neighbouring frames' mean [default: 0.1]\n"
"      --zero B             Per-step pull towards no displacement [default: 0.002]\n"
"      --max-disp M         Clamp of |delta| per axis [default: 0.2]\n"
"      --freeze-layers L    Comma-separated cage layer indices whose vertices keep zero displacement (e.g. 0, the\n"
"                           body: its fit mostly chases view-dependent label boundaries) [default: none]\n"
"      --alt-binding F      Dual binding file (gpu/cage.h): listed splats blend two candidate triangles by a weight\n"
"      --fit-binding        Learn those weights from the video (Adam on their logits); writes <output>/alt_binding.bin\n"
"      --binding-lr L       Step on the logits [default: 0.05]\n"
"      --fit-appearance M   Learn the opacity and SH colour of the splats flagged in M (uint8 per splat, the ply's order)\n"
"                           from the video; geometry stays frozen. Writes <output>/scene.ply\n"
"      --fit-splats M       Like --fit-appearance, and also the flagged splats' means, rotations and scales, their\n"
"                           gradients turned back into the canonical frame through their triangle's rotation\n"
"      --appearance-sh-lr L Step on the SH bands above 0 (0: keep them; b2crig borrows them from neighbours) [default: 0]\n"
"      --no-delta           Keep the cage displacement at zero (with --fit-binding: learn the binding only)\n"
"      --step-clip K        Clamp of a vertex's step length at K * lr: the shared second moment shrinks as the fit\n"
"                           converges, so without it a vertex with a persistently large gradient runs away [default: 4]\n"
"      --photo-weight W     Photometric (L1/SSIM) weight [default: 0.3]\n"
"      --alpha-weight W     Alpha-lane weight [default: 1.0]\n"
"      --label-weight W     Group-label weight, 0 = off [default: 1.0]\n"
"      --groups G           Three label groups, ';'-separated lists of Sapiens2 class ids (b2crig passes its own,\n                           e.g. 4;23,1;13 = hair | upper clothing + apparel | lower clothing); needed by the label term\n"
"      --cage-min-conf C    Splats with seg_conf below this bind across layers [default: 0.5]\n"
"      --max-resolution N   [default: 1920]\n"
"      --no-res-schedule    Full resolution from the first step\n"
"      --device N  --seed N\n";

std::vector<int> parse_ints(const std::string& s) { std::vector<int> v; size_t p = 0; while (p <= s.size()) { size_t e = s.find(',', p); if (e == std::string::npos) e = s.size(); if (e > p) v.push_back(atoi(s.substr(p, e - p).c_str())); p = e + 1; } return v; }

FitArgs parse(int argc, char** argv) {
  FitArgs a;
  for (int i = 2; i < argc; i++) {
    std::string k = argv[i];
    auto val = [&]() -> std::string { if (i + 1 >= argc) fail("a value is required for '%s'", k.c_str()); return argv[++i]; };
    auto fl = [&]() { return (float)atof(val().c_str()); };
    if (k == "--splat") a.splat = val();
    else if (k == "--cage") a.cage = val();
    else if (k == "--dataset") a.dataset = val();
    else if (k == "--output") a.output = val();
    else if (k == "--iters") a.iters = (uint32_t)atoi(val().c_str());
    else if (k == "--lr") a.lr = fl();
    else if (k == "--lap") a.lap = fl();
    else if (k == "--temporal") a.temporal = fl();
    else if (k == "--grad-smooth") a.grad_smooth = atoi(val().c_str());
    else if (k == "--zero") a.zero = fl();
    else if (k == "--max-disp") a.max_disp = fl();
    else if (k == "--step-clip") a.step_clip = fl();
    else if (k == "--alt-binding") a.alt_binding = val();
    else if (k == "--fit-binding") a.fit_binding = true;
    else if (k == "--binding-lr") a.binding_lr = fl();
    else if (k == "--no-delta") a.no_delta = true;
    else if (k == "--fit-appearance") a.appearance_mask = val();
    else if (k == "--fit-splats") { a.appearance_mask = val(); a.fit_splats = true; }
    else if (k == "--appearance-sh-lr") a.app_lr_sh = fl();
    else if (k == "--freeze-layers") {
      std::string v = argv[++i]; size_t p0 = 0;
      while (p0 <= v.size()) { size_t p1 = v.find(',', p0); if (p1 == std::string::npos) p1 = v.size(); if (p1 > p0) a.freeze_layers.push_back(std::stoi(v.substr(p0, p1 - p0))); p0 = p1 + 1; }
    }
    else if (k == "--photo-weight") a.photo_weight = fl();
    else if (k == "--alpha-weight") a.alpha_weight = fl();
    else if (k == "--label-weight") a.label_weight = fl();
    else if (k == "--cage-min-conf") a.min_conf = fl();
    else if (k == "--groups") { std::string v = val(); a.groups.clear(); size_t p = 0; while (p <= v.size()) { size_t e = v.find(';', p); if (e == std::string::npos) e = v.size(); a.groups.push_back(parse_ints(v.substr(p, e - p))); p = e + 1; } if (a.groups.size() != 3) fail("--groups needs exactly three ';'-separated groups"); }
    else if (k == "--max-resolution") a.max_resolution = (uint32_t)atoi(val().c_str());
    else if (k == "--no-res-schedule") a.res_schedule = false;
    else if (k == "--device") a.device = atoi(val().c_str());
    else if (k == "--seed") a.seed = (uint64_t)atoll(val().c_str());
    else if (k == "-h" || k == "--help") { fputs(HELP, stdout); exit(0); }
    else fail("unexpected argument '%s' found\n\nFor more information, try '--help'.", k.c_str());
  }
  if (a.splat.empty() || a.cage.empty() || a.dataset.empty() || a.output.empty()) fail("fit-cage needs --splat, --cage, --dataset and --output");
  if (a.fit_binding && a.alt_binding.empty()) fail("--fit-binding needs --alt-binding");
  return a;
}

// Label loss on the group feature render: L2 between the accumulated one-hot (ctx.out_feat.xyz) and the frame's
// one-hot of the same groups (classes outside every group, and background, are all-zero targets). Gradient into
// v_feat (scaled by grad_scale for the tensor-core backward); v_out is zeroed so the pass moves geometry only
// through the feature.
__global__ void label_loss_kernel(int n_px, int W, int H, int lW, int lH, const float4* __restrict__ feat, const uint8_t* __restrict__ labels, const int* __restrict__ lut,
                                  float w, float grad_scale, float4* __restrict__ v_feat, float4* __restrict__ v_out, float* __restrict__ accum) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_px) return;
  // labels live at the dataset's full resolution; sample the nearest label pixel for pyramid levels
  int x = i % W, y = i / W;
  int lx = min((int)((x + 0.5f) * lW / W), lW - 1), ly = min((int)((y + 0.5f) * lH / H), lH - 1);
  int g = lut[labels[(size_t)ly * lW + lx]];
  float t[3] = {g == 0 ? 1.f : 0.f, g == 1 ? 1.f : 0.f, g == 2 ? 1.f : 0.f};
  float4 f = feat[i];
  float d0 = f.x - t[0], d1 = f.y - t[1], d2 = f.z - t[2];
  float s = w / (float)n_px;
  v_feat[i] = make_float4(2.f * s * d0 * grad_scale, 2.f * s * d1 * grad_scale, 2.f * s * d2 * grad_scale, 0.f);
  v_out[i] = make_float4(0.f, 0.f, 0.f, 0.f);
  atomicAdd(accum, s * (d0 * d0 + d1 * d1 + d2 * d2));
}

__global__ void scatter_kernel(int n, const float3* __restrict__ g_pos, const int* __restrict__ bf, const float2* __restrict__ bb, const int3* __restrict__ faces,
                               float3* __restrict__ g_vert, float* __restrict__ vis) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int f = bf[i]; if (f < 0) return;
  float3 g = g_pos[i];
  if (g.x == 0.f && g.y == 0.f && g.z == 0.f) return;
  int3 t = faces[f]; float2 b = bb[i];
  const int vs[3] = {t.x, t.y, t.z}; const float ws[3] = {1.f - b.x - b.y, b.x, b.y};
  for (int k = 0; k < 3; k++) {
    if (ws[k] <= 0.f) continue;
    atomicAdd(&g_vert[vs[k]].x, ws[k] * g.x); atomicAdd(&g_vert[vs[k]].y, ws[k] * g.y); atomicAdd(&g_vert[vs[k]].z, ws[k] * g.z);
    if (vis) atomicAdd(&vis[vs[k]], ws[k]);
  }
}

__global__ void sumsq_kernel(int nv, const float3* __restrict__ g, float* __restrict__ out) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  float v = 0.f;
  if (k < nv) { float3 x = g[k]; v = x.x * x.x + x.y * x.y + x.z * x.z; }
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0 && v != 0.f) atomicAdd(out, v);
}

// Adam with ONE second moment for every vertex (v_glob[0], an EMA of the mean squared gradient over the vertices that
// received any): a vertex's step is proportional to its own gradient, so the weakly seen ones barely move instead of
// taking the full normalised step per-vertex Adam would give their noise. v_glob[1] holds this step's sum. The step
// length is clamped at step_clip * lr per vertex: the shared moment shrinks as the fit converges, and a vertex whose
// gradient stays large (a face vertex on a label edge, its splats stretched by its own displacement) would otherwise
// take ever larger steps - the divergence seen past ~150 steps per frame.
__global__ void adam_kernel(int nv, const float3* __restrict__ g, float3* __restrict__ d, float3* __restrict__ m, float* __restrict__ v_glob, int n_seen,
                            float lr, float zero, float max_disp, float step_clip, int t, int t_glob) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nv) return;
  const float b1 = 0.9f, b2 = 0.999f;
  float vv = v_glob[0];
  if (n_seen > 0) vv = b2 * vv + (1.f - b2) * v_glob[1] / (3.f * (float)n_seen);
  float3 gg = g[k], dd = d[k], mm = m[k];
  mm.x = b1 * mm.x + (1.f - b1) * gg.x; mm.y = b1 * mm.y + (1.f - b1) * gg.y; mm.z = b1 * mm.z + (1.f - b1) * gg.z;
  m[k] = mm;
  float bc1 = 1.f / (1.f - powf(b1, (float)t)), bc2 = 1.f / (1.f - powf(b2, (float)t_glob));
  float den = sqrtf(vv * bc2) + 1e-20f;
  if (vv > 0.f) {
    float3 st = mm * (lr * bc1 / den);
    const float len = sqrtf(st.x * st.x + st.y * st.y + st.z * st.z), cap = step_clip * lr;
    if (step_clip > 0.f && len > cap) st = st * (cap / len);
    dd.x -= st.x; dd.y -= st.y; dd.z -= st.z;
  }
  dd = dd * (1.f - zero);
  dd.x = fminf(fmaxf(dd.x, -max_disp), max_disp); dd.y = fminf(fmaxf(dd.y, -max_disp), max_disp); dd.z = fminf(fmaxf(dd.z, -max_disp), max_disp);
  d[k] = dd;
}

__global__ void vglob_commit(float* v_glob, int n_seen) {
  if (n_seen > 0) v_glob[0] = 0.999f * v_glob[0] + 0.001f * v_glob[1] / (3.f * (float)n_seen);
  v_glob[1] = 0.f;
}

__global__ void count_seen_kernel(int nv, const float3* __restrict__ g, int* __restrict__ out) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < nv) { float3 x = g[k]; if (x.x != 0.f || x.y != 0.f || x.z != 0.f) atomicAdd(out, 1); }
}

__global__ void prox_kernel(int nv, const float3* __restrict__ d, const int* __restrict__ nb_off, const int* __restrict__ nb, const float3* __restrict__ prev, const float3* __restrict__ next,
                            float lap, float temporal, float3* __restrict__ out) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nv) return;
  float3 x = d[k];
  int a = nb_off[k], e = nb_off[k + 1];
  if (lap > 0.f && e > a) {
    float3 s = make_float3(0.f, 0.f, 0.f);
    for (int j = a; j < e; j++) s = s + d[nb[j]];
    s = s * (1.f / (float)(e - a));
    x = x * (1.f - lap) + s * lap;
  }
  if (temporal > 0.f && (prev || next)) {
    float3 p = prev ? prev[k] : next[k], q = next ? next[k] : prev[k];
    x = x * (1.f - temporal) + (p + q) * (0.5f * temporal);
  }
  out[k] = x;
}

// Gradient of the loss w.r.t. the dual-binding logits: the splat's positional gradient along pos_B - pos_A, times
// dw/dlogit; then Adam on the logits and w = sigmoid(logit).
__global__ void alt_step_kernel(int na, const int* __restrict__ idx, const float3* __restrict__ g1, const float3* __restrict__ g2,
                                const float3* __restrict__ dpos, float* __restrict__ logit, float* __restrict__ m, float* __restrict__ v,
                                float* __restrict__ w, float lr, int t) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= na) return;
  int i = idx[j]; float3 g = g1[i]; if (g2) g = g + g2[i];
  float3 d = dpos[j]; float ww = w[j];
  float gl = (g.x * d.x + g.y * d.y + g.z * d.z) * ww * (1.f - ww);
  if (gl == 0.f) return;   // not seen this step
  const float b1 = 0.9f, b2 = 0.999f;
  m[j] = b1 * m[j] + (1.f - b1) * gl; v[j] = b2 * v[j] + (1.f - b2) * gl * gl;
  float mh = m[j] / (1.f - powf(b1, (float)t)), vh = v[j] / (1.f - powf(b2, (float)t));
  logit[j] = fminf(fmaxf(logit[j] - lr * mh / (sqrtf(vh) + 1e-12f), -8.f), 8.f);
  w[j] = 1.f / (1.f + expf(-logit[j]));
}

}  // namespace

int fit_cage_main(int argc, char** argv) {
  FitArgs a = parse(argc, argv);
  CUDA_CHECK(cudaSetDevice(a.device));
  cudaStream_t stream = 0;
  SplatCloud cloud = read_ply(a.splat);
  if (!cloud.has_labels) log_warn("fit-cage: %s has no seg_label; every splat binds across layers", a.splat.c_str());
  Model model; model.upload(cloud);
  CageRig cage;
  if (!cage.load(a.cage)) fail("--cage %s: cannot open", a.cage.c_str());
  cage.bind(model, cloud.has_labels ? cloud.labels : std::vector<float>{}, a.min_conf, stream);
  cage.enable_delta(stream);
  DevBuf<uint8_t> app_mask; int app_t = 0;
  if (!a.appearance_mask.empty()) {
    std::ifstream mf(a.appearance_mask, std::ios::binary);
    if (!mf) fail("--fit-appearance %s: cannot open", a.appearance_mask.c_str());
    std::vector<uint8_t> mh((size_t)model.n);
    if (!mf.read((char*)mh.data(), (std::streamsize)mh.size())) fail("--fit-appearance %s: expected %d bytes (one per splat)", a.appearance_mask.c_str(), model.n);
    size_t nm = 0; for (auto x : mh) nm += x != 0;
    app_mask.upload(mh, stream);
    log_info("fit-cage: learning the appearance of %zu/%d splats", nm, model.n);
  }
  std::vector<int> alt_s, alt_fb; std::vector<float> alt_w0;
  DevBuf<float> alt_logit, alt_m, alt_v; int alt_t = 0;
  if (!a.alt_binding.empty()) {
    if (!load_alt_binding(a.alt_binding, alt_s, alt_fb, alt_w0)) fail("--alt-binding %s: cannot open", a.alt_binding.c_str());
    cage.bind_alt(model, alt_s, alt_fb, alt_w0, stream);
    std::vector<float> lg(alt_w0.size());
    for (size_t j = 0; j < lg.size(); j++) { float w = std::min(std::max(alt_w0[j], 1e-3f), 1.f - 1e-3f); lg[j] = std::log(w / (1.f - w)); }
    alt_logit.upload(lg, stream); alt_m.reserve(lg.size()); alt_m.zero(stream); alt_v.reserve(lg.size()); alt_v.zero(stream);
    log_info("fit-cage: dual binding for %d splats%s", cage.n_alt, a.fit_binding ? " (learning the weights)" : "");
  }

  Config dc; dc.source = a.dataset; dc.max_resolution = a.max_resolution; dc.alpha_mode = AlphaMode::Transparent;
  Dataset ds = load_dataset(dc);
  if (ds.train.empty()) fail("fit-cage: the dataset has no frames");
  std::vector<int> frame_of(ds.train.size(), -1);
  int matched = 0;
  for (size_t i = 0; i < ds.train.size(); i++) {
    std::string nm = ds.train[i].name; int fr = cage.frame_index(nm);
    if (fr < 0) { auto d0 = nm.rfind('.'); if (d0 != std::string::npos) fr = cage.frame_index(nm.substr(0, d0)); }
    if (fr < 0) { auto d0 = nm.find('@'); if (d0 != std::string::npos) fr = cage.frame_index(nm.substr(0, d0)); }
    frame_of[i] = fr; matched += fr >= 0;
  }
  if (!matched) fail("fit-cage: no dataset image is named like a cage frame");
  for (int L : a.freeze_layers) if (L < 0 || L >= (int)cage.layers.size()) fail("fit-cage: --freeze-layers %d: the cage has %zu layers", L, cage.layers.size());
  std::vector<int> frames_used; for (size_t i = 0; i < ds.train.size(); i++) if (frame_of[i] >= 0) frames_used.push_back((int)i);
  const bool use_labels = a.label_weight > 0.f && ds.n_labels > 0 && cloud.has_labels;
  if (use_labels && a.groups.size() != 3) fail("fit-cage: the label term needs --groups (three ';'-separated class lists; b2crig passes them), or --label-weight 0");
  if (a.label_weight > 0.f && !use_labels) log_warn("fit-cage: the label term is off (the dataset has no labels/ sidecar or the splat no seg_label)");
  GpuViews gv; gv.upload(ds.train);
  gv.build_pyramid(a.res_schedule ? 3 : 1, stream);
  log_info("fit-cage: %d splats, cage %zu layers / %d verts / %d frames; %d/%zu dataset frames matched; labels %s",
           model.n, cage.layers.size(), cage.nv, cage.nframes, matched, ds.train.size(), use_labels ? "on" : "off");

  // Group feature per splat and the class -> group table.
  std::vector<int> lut_h(256, -1);
  for (int g = 0; g < 3; g++) for (int c : a.groups[g]) if (c >= 0 && c < 256) lut_h[c] = g;
  DevBuf<int> lut; lut.upload(lut_h, stream);
  DevBuf<float> group_feat;
  if (use_labels) {
    std::vector<float> h((size_t)model.n * 3, 0.f);
    for (int i = 0; i < model.n; i++) { int c = (int)std::lround(cloud.labels[(size_t)i * 2]); if (c >= 0 && c < 256 && lut_h[c] >= 0) h[(size_t)i * 3 + lut_h[c]] = 1.f; }
    group_feat.upload(h, stream);
  }
  // Mesh neighbours (CSR) for the spatial pull.
  auto fc = cage.faces.download(cage.nf, stream);
  std::vector<std::vector<int>> nbr(cage.nv);
  for (auto& t : fc) { int v[3] = {t.x, t.y, t.z}; for (int p = 0; p < 3; p++) for (int q = 0; q < 3; q++) if (p != q) nbr[v[p]].push_back(v[q]); }
  std::vector<int> nb_off(cage.nv + 1, 0), nb_h;
  for (int k = 0; k < cage.nv; k++) { auto& L = nbr[k]; std::sort(L.begin(), L.end()); L.erase(std::unique(L.begin(), L.end()), L.end()); nb_off[k + 1] = nb_off[k] + (int)L.size(); nb_h.insert(nb_h.end(), L.begin(), L.end()); }
  DevBuf<int> d_nb_off, d_nb; d_nb_off.upload(nb_off, stream); d_nb.upload(nb_h, stream);

  const size_t NV = (size_t)cage.nv;
  DevBuf<float3> g_pos, g_pos2, g_vert, m_d, tmp; DevBuf<float> v_d, vis; DevBuf<int> n_seen_d; n_seen_d.reserve(1);
  int t_glob = 0;
  g_pos.reserve(model.n); g_pos2.reserve(model.n); g_vert.reserve(NV); tmp.reserve(NV);
  m_d.reserve(cage.nframes * NV); m_d.zero(stream); v_d.reserve(2); v_d.zero(stream);
  vis.reserve(cage.nframes * NV); vis.zero(stream);
  std::vector<int> t_frame(cage.nframes, 0);
  RenderCtx ctx;
  std::mt19937_64 rng(a.seed);

  auto render_frame = [&](int vi, int level, RenderParams& rp) -> const ViewGPU& {
    const ViewGPU& view = (level > 0 && !gv.lvl_views[level].empty()) ? gv.lvl_views[level][vi] : gv.views[vi];
    rp.cam = CameraGPU::from(gv.cams[vi], view.W, view.H);
    cage.pose(frame_of[vi], model, stream);
    rp.pos_override = cage.pos_view; rp.quat_override = cage.quat_view; rp.lscale_override = cage.lscale_view; rp.sh_frame = cage.sh_frame;
    rp.sh_degree = model.degree; rp.bwd_info = true;
    if (view.W != ctx.W || view.H != ctx.H) ctx.setup(view.W, view.H, model.cap, stream);
    return view;
  };
  // One loss evaluation (and, with `grad`, the per-vertex gradient into g_vert) of dataset view vi.
  auto step_view = [&](int vi, int level, bool grad, float* photo_loss, float* label_loss) {
    RenderParams rp;
    const ViewGPU& view = render_frame(vi, level, rp);
    const float grad_scale = 3.f * (float)view.W * (float)view.H;
    render_forward(ctx, model, rp, stream);
    ctx.loss_accum.zero(stream, 4);
    LossParams lp;
    lp.l1_w = (1.f - a.ssim_weight) * a.photo_weight; lp.ssim_w = -a.ssim_weight * a.photo_weight;
    lp.composite = view.has_alpha; lp.alpha_lane = view.has_alpha && a.alpha_weight > 0.f; lp.match_alpha_weight = a.alpha_weight;
    lp.grad_scale = grad_scale;
    photometric_loss(ctx, view, lp, stream);
    if (photo_loss) *photo_loss = ctx.loss_accum.download(1, stream)[0];
    OptimParams op; op.cam = rp.cam; op.active_sh_degree = rp.sh_degree; op.t = 1; op.frozen = true;
    const bool app = grad && app_mask.ptr != nullptr;
    if (app) {   // appearance: opacity and colour of the masked splats, geometry frozen
      op.frozen = false; op.update_mask = app_mask.ptr; op.t = ++app_t;
      op.lr_mean = 0.f; op.lr_rot = 0.f; op.lr_scale = 0.f;
      if (a.fit_splats) { op.lr_mean = a.sp_lr_mean; op.lr_rot = a.sp_lr_rot; op.lr_scale = a.sp_lr_scale; op.canon_rot = cage.sh_frame.ptr; }
      op.lr_dc = a.app_lr_dc; op.lr_opac = a.app_lr_opac; op.lr_sh_rest = a.app_lr_sh;
    }
    op.pos_override = rp.pos_override; op.quat_override = rp.quat_override; op.lscale_override = rp.lscale_override; op.sh_frame = rp.sh_frame;
    if (grad) {
      rasterize_backward_tc(ctx, model, rp, grad_scale, stream);
      op.g_pos_out = g_pos; optimizer_step(ctx, model, op, stream);
    }
    if (use_labels && gv.views[vi].labels) {
      RenderParams rq = rp; rq.feat = FeatureMode::Buffer; rq.feat_buffer = group_feat;
      render_forward(ctx, model, rq, stream);
      ctx.loss_accum.zero(stream, 4);
      const int n_px = view.W * view.H;
      label_loss_kernel<<<div_up(n_px, 256), 256, 0, stream>>>(n_px, view.W, view.H, gv.views[vi].W, gv.views[vi].H, ctx.out_feat, gv.views[vi].labels, lut, a.label_weight,
                                                               grad_scale, ctx.v_feat, ctx.v_out, ctx.loss_accum);
      CUDA_KERNEL_CHECK();
      if (label_loss) *label_loss = ctx.loss_accum.download(1, stream)[0];
      if (grad) {
        rasterize_backward_tc(ctx, model, rq, grad_scale, stream);
        op.frozen = true; op.update_mask = nullptr;   // the label pass only feeds the geometry
        op.g_pos_out = g_pos2; optimizer_step(ctx, model, op, stream);
      }
    } else if (label_loss) *label_loss = 0.f;
  };

  // Losses before the fit.
  std::vector<float> photo0(ds.train.size(), 0.f), label0(ds.train.size(), 0.f), photo1(ds.train.size(), 0.f), label1(ds.train.size(), 0.f);
  for (int vi : frames_used) step_view(vi, 0, false, &photo0[vi], &label0[vi]);

  double t0 = now_seconds();
  for (uint32_t it = 1; it <= a.iters; it++) {
    int vi = frames_used[rng() % frames_used.size()];
    int fr = frame_of[vi];
    float pr = (float)it / (float)a.iters;
    int level = a.res_schedule ? (pr < 0.3f ? 2 : (pr < 0.6f ? 1 : 0)) : 0;
    g_pos.zero(stream); g_pos2.zero(stream); g_vert.zero(stream);
    step_view(vi, level, true, nullptr, nullptr);
    if (a.fit_splats) cage.refresh_offsets(model, stream);
    if (a.fit_binding && cage.n_alt > 0) {
      alt_step_kernel<<<div_up(cage.n_alt, 256), 256, 0, stream>>>(cage.n_alt, cage.alt_idx, g_pos, use_labels ? g_pos2.ptr : nullptr, cage.alt_dpos,
                                                                   alt_logit, alt_m, alt_v, cage.alt_w, a.binding_lr, ++alt_t);
      CUDA_KERNEL_CHECK();
    }
    if (a.no_delta) continue;
    float* vis_f = level == 0 ? vis.ptr + (size_t)fr * NV : nullptr;
    scatter_kernel<<<div_up(model.n, 256), 256, 0, stream>>>(model.n, g_pos, cage.bind_f, cage.bind_b, cage.faces, g_vert, vis_f);
    if (use_labels) scatter_kernel<<<div_up(model.n, 256), 256, 0, stream>>>(model.n, g_pos2, cage.bind_f, cage.bind_b, cage.faces, g_vert, nullptr);
    CUDA_KERNEL_CHECK();
    for (int s2 = 0; s2 < a.grad_smooth; s2++) {
      prox_kernel<<<div_up(cage.nv, 256), 256, 0, stream>>>(cage.nv, g_vert, d_nb_off, d_nb, nullptr, nullptr, 0.5f, 0.f, tmp);
      CUDA_CHECK(cudaMemcpyAsync(g_vert.ptr, tmp.ptr, NV * sizeof(float3), cudaMemcpyDeviceToDevice, stream));
    }
    float3* d = cage.delta.ptr + (size_t)fr * NV;
    n_seen_d.zero(stream);
    count_seen_kernel<<<div_up(cage.nv, 256), 256, 0, stream>>>(cage.nv, g_vert, n_seen_d);
    sumsq_kernel<<<div_up(cage.nv, 256), 256, 0, stream>>>(cage.nv, g_vert, v_d.ptr + 1);
    const int n_seen = n_seen_d.download(1, stream)[0];
    adam_kernel<<<div_up(cage.nv, 256), 256, 0, stream>>>(cage.nv, g_vert, d, m_d.ptr + (size_t)fr * NV, v_d, n_seen, a.lr, a.zero, a.max_disp, a.step_clip, ++t_frame[fr], ++t_glob);
    vglob_commit<<<1, 1, 0, stream>>>(v_d, n_seen);
    const float3* prev = fr > 0 ? cage.delta.ptr + (size_t)(fr - 1) * NV : nullptr;
    const float3* next = fr + 1 < cage.nframes ? cage.delta.ptr + (size_t)(fr + 1) * NV : nullptr;
    prox_kernel<<<div_up(cage.nv, 256), 256, 0, stream>>>(cage.nv, d, d_nb_off, d_nb, prev, next, a.lap, a.temporal, tmp);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaMemcpyAsync(d, tmp.ptr, NV * sizeof(float3), cudaMemcpyDeviceToDevice, stream));
    for (int L : a.freeze_layers)
      CUDA_CHECK(cudaMemsetAsync(d + cage.layers[L].v_off, 0, (size_t)cage.layers[L].v_count * sizeof(float3), stream));
    if (it % 1000 == 0 || it == a.iters) {
      auto dl = cage.delta.download(cage.nframes * NV, stream);
      double mx = 0, sum = 0; for (auto& x : dl) { double l = std::sqrt((double)x.x * x.x + (double)x.y * x.y + (double)x.z * x.z); mx = std::max(mx, l); sum += l; }
      log_info("fit-cage iter %u/%u (level %d): |delta| mean %.2f mm, max %.1f mm, %.1f it/s", it, a.iters, level, 1000.0 * sum / dl.size(), 1000.0 * mx, it / (now_seconds() - t0));
    }
  }
  for (int vi : frames_used) step_view(vi, 0, false, &photo1[vi], &label1[vi]);

  fs::create_directories(a.output);
  auto dl = cage.delta.download(cage.nframes * NV, stream);
  auto vh = vis.download(cage.nframes * NV, stream);
  { std::ofstream o(fs::path(a.output) / "delta.f32", std::ios::binary); o.write((const char*)dl.data(), dl.size() * sizeof(float3)); }
  { std::ofstream o(fs::path(a.output) / "vis.f32", std::ios::binary); o.write((const char*)vh.data(), vh.size() * sizeof(float)); }
  if (app_mask.ptr) {
    SplatCloud c = model.download(stream);
    if (cloud.has_labels) { c.labels = cloud.labels; c.has_labels = true; }
    write_ply((fs::path(a.output) / "scene.ply").string(), c, {"Exported from Brush", "Vertical axis: y", format("SH degree: %d", c.sh_degree), "SplatRenderMode: default"});
  }
  if (cage.n_alt > 0 && a.fit_binding) {
    auto wl = cage.alt_w.download(cage.n_alt, stream);
    save_alt_binding((fs::path(a.output) / "alt_binding.bin").string(), alt_s, alt_fb, wl);
    int nb = 0; for (float w : wl) nb += w > 0.5f;
    log_info("fit-cage: dual binding: %d/%d splats prefer triangle B", nb, cage.n_alt);
  }
  nlohmann::json fr_j = nlohmann::json::array();
  double p0 = 0, p1 = 0, l0 = 0, l1 = 0;
  for (int vi : frames_used) {
    fr_j.push_back({{"name", ds.train[vi].name}, {"frame", frame_of[vi]}, {"photo_before", photo0[vi]}, {"photo_after", photo1[vi]}, {"label_before", label0[vi]}, {"label_after", label1[vi]}});
    p0 += photo0[vi]; p1 += photo1[vi]; l0 += label0[vi]; l1 += label1[vi];
  }
  const double nfu = (double)frames_used.size();
  nlohmann::json out = {{"frames", cage.names}, {"n_verts", cage.nv}, {"iters", a.iters}, {"views", fr_j},
                        {"photo_before", p0 / nfu}, {"photo_after", p1 / nfu}, {"label_before", l0 / nfu}, {"label_after", l1 / nfu}};
  std::ofstream(fs::path(a.output) / "fit.json") << out.dump(1);
  log_info("fit-cage: photometric %.5f -> %.5f, labels %.5f -> %.5f (per-frame means); wrote %s", p0 / nfu, p1 / nfu, l0 / nfu, l1 / nfu, a.output.c_str());
  return 0;
}

}  // namespace b2c
