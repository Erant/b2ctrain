#include "render/renderer.h"
#include "render/confidence.h"
#include "gpu/render.h"
#include "gpu/cage.h"
#include "gpu/cage_app.h"
#include "gpu/cage_open.h"
#include "train/evidence.h"
#include "train/gpu_views.h"
#include "dataset/views.h"
#include "ply.h"
#include "cli.h"
#include "util/log.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "json.hpp"
#include <fstream>
#include <filesystem>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace b2c {
namespace fs = std::filesystem;

namespace {
struct RenderArgs {
  std::string splat, cameras, output_dir = "out", output_format = "png";
  float background[3] = {1.f, 1.f, 1.f};
  bool confidence = false;
  float cull[3] = {0.5f, 0.5f, 0.5f};
  float gate_lo = 0.45f, gate_hi = 0.65f;
  bool sidecar = false;
  std::string dataset, write_evidence;
  float evidence_normal_weight = 0.f;
  ConfidenceParams conf;
  Config ds_cfg;  // dataset options for --dataset
  int device = 0;
  int sh_degree = -1;  // bands to evaluate; -1 = every band the ply carries
  // Cage deformation (gpu/cage.h): each camera renders the cage frame of the same name.
  std::string cage, alt_binding, export_posed, export_frame, cage_app, export_binding;
  bool cage_app_debug = false;
  bool cage_open_off = false, cage_open_debug = false;   // two-state splats (ply open_*, gpu/cage_open.h): ignore them / show g
  float cage_open_start = 0.f, cage_open_end = 0.f;      // override the ply header's gate range (0 = keep)
  float cage_min_conf = 0.5f;
  float cage_max_growth = 0.f;
  float cage_fade_start = 0.f, cage_fade_end = 0.f;
  float cage_fill_start = 0.f, cage_fill_end = 0.f;   // filler splats (ply cage_fill) fade in over this stretch range; the fade's when unset
  std::vector<int> class_mask;   // write <stem>.mask.png: the accumulated alpha of splats with these seg_labels
  bool label_maps = false;       // write <stem>.labels.png: per pixel the seg_label with the most weight (0 where alpha < 0.5)
};

const char* HELP =
"Render a trained splat .ply against an explicit camera list\n\n"
"Usage: b2ctrain render [OPTIONS] --splat <SPLAT> --cameras <CAMERAS>\n\n"
"Options:\n"
"      --splat <SPLAT>                  Trained gaussian splat scene, as .ply\n"
"      --cameras <CAMERAS>              Camera list, in body2colmap.Camera pixel-space terms (cameras.json)\n"
"      --output-dir <OUTPUT_DIR>        Directory to write one RGBA image per camera into [default: out]\n"
"      --background <BACKGROUND>        Background color composited under the splat's accumulated alpha, as \"r,g,b\" in 0..1. Ignored with --confidence [default: 1.0,1.0,1.0]\n"
"      --output-format <OUTPUT_FORMAT>  Image encoder (png only) [default: png]\n"
"      --device <N>                     CUDA device [default: 0]\n"
"      --sh-degree <N>                  Highest spherical-harmonic band to evaluate, 0..3; 0 renders the DC colour only. Clamped to the ply's own degree [default: the ply's degree]\n\n"
"Cage options (animating a subject; see gpu/cage.h):\n"
"      --cage <CAGE>                    Deform the splat by this cage file; each camera renders the cage frame named like it\n"
"      --export-posed <PLY>             Write the splat as posed by the cage frame --export-frame (means, rotations, scales;\n"
"                                       SH left in each splat's canonical frame) and <PLY>.frame.f32: per splat the rotation\n"
"                                       (w,x,y,z) its triangle applied, to rotate the SH with; then render as usual\n"
"      --export-frame <NAME>            The cage frame --export-posed uses\n"
"      --export-binding <FILE>          Write every splat's cage binding (B2CBIND1: char[8] magic; int32 n; int32 face[n];\n"
"                                       float2 barycentrics[n]; float3 offset[n], in the canonical triangle frame / its size)\n"
"                                       for exporters (b2crig's glTF). Without --cameras, nothing is rendered\n"
"      --alt-binding <F>                Dual binding file (gpu/cage.h; fit-cage --fit-binding writes the learned one)\n"
"      --cage-min-conf <C>              Splats with seg_conf below this bind to the nearest triangle of any layer [default: 0.5]\n"
"      --cage-max-growth <G>            Bound splats follow their triangle's size change (offset and scales) only within 1/G..G [default: off]\n"
"      --cage-fade-start <S>            Stretch fade: bound splats' opacity falls linearly to 0 as their triangle grows from S\n"
"                                       to --cage-fade-end x its canonical size (an armpit membrane as the arm lifts) [default: off]\n"
"      --cage-fade-end <E>              Stretch fade end ratio [default: off]\n"
"      --cage-fill-start <S>            Filler splats (ply cage_fill > 0) fade IN as their triangle grows from S to --cage-fill-end\n"
"                                       x its canonical size [default: the fade range]\n"
"      --cage-fill-end <E>              Filler fade-in end ratio [default: the fade end]\n"
"      --cage-app <APP>                 Pose-dependent appearance MLP (gpu/cage_app.h; the .app a --cage-app training exported)\n"
"      --cage-app-debug                 Show the MLP's blend weight: splats turn magenta by |alpha|\n"
"      --cage-open-off                  Ignore the ply's two-state open_* properties (gpu/cage_open.h): render the plain splat\n"
"      --cage-open-debug                Show the open state's gate: splats turn magenta by g\n"
"      --cage-open-start <S>            Override the ply header's open-state gate range start (see --cage-open in train)\n"
"      --cage-open-end <E>              Override the ply header's open-state gate range end\n"
"      --class-mask <C,C,...>           Also write <name>.mask.png: the accumulated alpha of the splats whose seg_label is listed\n"
"      --label-maps                     Also write <name>.labels.png: per pixel the seg_label carrying the most weight, 0 where alpha < 0.5\n\n"
"Confidence options:\n"
"      --confidence                     Gate every pixel by the per-splat multi-view confidence\n"
"      --cull-color <CULL_COLOR>        Colour culled pixels resolve to and the compositing background [default: 0.5,0.5,0.5]\n"
"      --gate-lo <GATE_LO>              Per-pixel confidence at or below which a pixel is fully culled [default: 0.45]\n"
"      --gate-hi <GATE_HI>              Per-pixel confidence at or above which a pixel is fully kept [default: 0.65]\n"
"      --confidence-sidecar             Also write the raw per-pixel confidence as <stem>.conf.<format>\n"
"      --dataset <DATASET>              Training dataset directory to measure evidence against when the ply carries none\n"
"      --evidence-normal-weight <W>     Weight of the normal-map residual in the evidence residual [default: 0.0]\n"
"      --write-evidence <PLY>           After measuring evidence from --dataset, write the splat with ev_* properties here\n"
"      --conf-tau <TAU>                 [default: 0.08]\n"
"      --conf-min-views <MIN_VIEWS>     Effective supporting views (ev_views, the participation ratio of the per-view in-mask mass) for full support [default: 4]\n"
"      --conf-inmask-lo <INMASK_LO>     [default: 0.3]\n"
"      --conf-inmask-hi <INMASK_HI>     [default: 0.8]\n"
"      --conf-angle-margin <DEG>        [default: 30]\n"
"      --conf-angle-soft <DEG>          [default: 15]\n"
"      --conf-facing                    Also distrust disc-like splats seen at grazing or back-facing angles\n"
"      --conf-graze-deg <DEG>           [default: 80]\n\n"
"Dataset Options (for --dataset):\n"
"      --max-frames <N>  --max-resolution <N> [default: 1920]  --eval-split-every <N>  --subsample-frames <N>  --subsample-points <N>\n"
"      --alpha-mode <masked|transparent>  --max-scene-batch-cache-size <SIZE>\n";

float parse_float(const std::string& v, const char* name) { char* e = nullptr; float f = strtof(v.c_str(), &e); if (e == v.c_str() || *e) fail("invalid value '%s' for '%s'", v.c_str(), name); return f; }
void parse_rgb(const std::string& v, float* out, const char* name) { if (sscanf(v.c_str(), "%f,%f,%f", &out[0], &out[1], &out[2]) != 3) fail("invalid %s '%s' (expected r,g,b)", name, v.c_str()); }

RenderArgs parse_render_args(int argc, char** argv) {
  RenderArgs a;
  for (int i = 2; i < argc; i++) {
    std::string k = argv[i];
    auto val = [&](const char* name) -> std::string { if (i + 1 >= argc) fail("a value is required for '%s'", name); return argv[++i]; };
    if (k == "--splat") a.splat = val("--splat");
    else if (k == "--cameras") a.cameras = val("--cameras");
    else if (k == "--output-dir") a.output_dir = val("--output-dir");
    else if (k == "--output-format") a.output_format = val("--output-format");
    else if (k == "--background") parse_rgb(val("--background"), a.background, "--background");
    else if (k == "--device") a.device = atoi(val("--device").c_str());
    else if (k == "--sh-degree") { a.sh_degree = atoi(val(k.c_str()).c_str()); if (a.sh_degree < 0 || a.sh_degree > 3) fail("invalid --sh-degree %d (expected 0..3)", a.sh_degree); }
    else if (k == "--cage") a.cage = val("--cage");
    else if (k == "--alt-binding") a.alt_binding = val("--alt-binding");
    else if (k == "--export-posed") a.export_posed = val("--export-posed");
    else if (k == "--export-binding") a.export_binding = val("--export-binding");
    else if (k == "--export-frame") a.export_frame = val("--export-frame");
    else if (k == "--cage-min-conf") a.cage_min_conf = parse_float(val(k.c_str()), k.c_str());
    else if (k == "--cage-max-growth") a.cage_max_growth = parse_float(val(k.c_str()), k.c_str());
    else if (k == "--cage-fade-start") a.cage_fade_start = parse_float(val(k.c_str()), k.c_str());
    else if (k == "--cage-fade-end") a.cage_fade_end = parse_float(val(k.c_str()), k.c_str());
    else if (k == "--cage-fill-start") a.cage_fill_start = parse_float(val(k.c_str()), k.c_str());
    else if (k == "--cage-fill-end") a.cage_fill_end = parse_float(val(k.c_str()), k.c_str());
    else if (k == "--cage-app") a.cage_app = val("--cage-app");
    else if (k == "--cage-app-debug") a.cage_app_debug = true;
    else if (k == "--cage-open-off") a.cage_open_off = true;
    else if (k == "--cage-open-debug") a.cage_open_debug = true;
    else if (k == "--cage-open-start") a.cage_open_start = parse_float(val(k.c_str()), k.c_str());
    else if (k == "--cage-open-end") a.cage_open_end = parse_float(val(k.c_str()), k.c_str());
    else if (k == "--class-mask") { std::string v = val(k.c_str()); size_t p0 = 0; while (p0 <= v.size()) { size_t e = v.find(',', p0); if (e == std::string::npos) e = v.size(); if (e > p0) a.class_mask.push_back(atoi(v.substr(p0, e - p0).c_str())); p0 = e + 1; } }
    else if (k == "--label-maps") a.label_maps = true;
    else if (k == "--confidence") a.confidence = true;
    else if (k == "--cull-color") parse_rgb(val("--cull-color"), a.cull, "--cull-color");
    else if (k == "--gate-lo") a.gate_lo = parse_float(val(k.c_str()), "--gate-lo");
    else if (k == "--gate-hi") a.gate_hi = parse_float(val(k.c_str()), "--gate-hi");
    else if (k == "--confidence-sidecar") a.sidecar = true;
    else if (k == "--dataset") a.dataset = val("--dataset");
    else if (k == "--evidence-normal-weight") a.evidence_normal_weight = parse_float(val(k.c_str()), k.c_str());
    else if (k == "--write-evidence") a.write_evidence = val("--write-evidence");
    else if (k == "--conf-tau") a.conf.tau = parse_float(val(k.c_str()), k.c_str());
    else if (k == "--conf-min-views") a.conf.min_views = parse_float(val(k.c_str()), k.c_str());
    else if (k == "--conf-inmask-lo") a.conf.inmask_lo = parse_float(val(k.c_str()), k.c_str());
    else if (k == "--conf-inmask-hi") a.conf.inmask_hi = parse_float(val(k.c_str()), k.c_str());
    else if (k == "--conf-angle-margin") a.conf.angle_margin_deg = parse_float(val(k.c_str()), k.c_str());
    else if (k == "--conf-angle-soft") a.conf.angle_soft_deg = parse_float(val(k.c_str()), k.c_str());
    else if (k == "--conf-facing") a.conf.facing = true;
    else if (k == "--conf-graze-deg") a.conf.graze_deg = parse_float(val(k.c_str()), k.c_str());
    else if (k == "--max-frames") a.ds_cfg.max_frames = (uint32_t)atoi(val(k.c_str()).c_str());
    else if (k == "--max-resolution") a.ds_cfg.max_resolution = (uint32_t)atoi(val(k.c_str()).c_str());
    else if (k == "--eval-split-every") a.ds_cfg.eval_split_every = (uint32_t)atoi(val(k.c_str()).c_str());
    else if (k == "--subsample-frames") a.ds_cfg.subsample_frames = (uint32_t)atoi(val(k.c_str()).c_str());
    else if (k == "--subsample-points") a.ds_cfg.subsample_points = (uint32_t)atoi(val(k.c_str()).c_str());
    else if (k == "--max-scene-batch-cache-size") val(k.c_str());
    else if (k == "--alpha-mode") { auto v = val(k.c_str()); if (v == "masked") a.ds_cfg.alpha_mode = AlphaMode::Masked; else if (v == "transparent") a.ds_cfg.alpha_mode = AlphaMode::Transparent; else fail("invalid --alpha-mode '%s'", v.c_str()); }
    else if (k == "-h" || k == "--help") { fputs(HELP, stdout); exit(0); }
    else fail("unexpected argument '%s' found\n\nFor more information, try '--help'.", k.c_str());
  }
  if (!a.export_binding.empty() && a.cage.empty()) fail("--export-binding needs --cage");
  if (a.splat.empty() || (a.cameras.empty() && a.export_binding.empty())) fail("the following required arguments were not provided: --splat <SPLAT> --cameras <CAMERAS>");
  if (a.output_format != "png") fail("only png output is supported");
  if (a.confidence && (!a.class_mask.empty() || a.label_maps)) fail("--confidence cannot be combined with --class-mask / --label-maps (they share the feature channel)");
  return a;
}

float smoothstep_gate(float lo, float hi, float x) {
  if (hi <= lo) return x >= lo ? 1.f : 0.f;
  float t = std::min(std::max((x - lo) / (hi - lo), 0.f), 1.f);
  return t * t * (3.f - 2.f * t);
}
unsigned char to_u8(float v) { return (unsigned char)std::lround(std::min(std::max(v, 0.f), 1.f) * 255.f); }
}  // namespace

Camera camera_from_json(const nlohmann::json& c, int W, int H) {
  Camera cam;
  cam.width = W; cam.height = H;
  cam.fx = c.at("fx").get<float>(); cam.fy = c.at("fy").get<float>(); cam.cx = c.at("cx").get<float>(); cam.cy = c.at("cy").get<float>();
  // rotation: OpenGL camera-to-world (columns = local axes, Y up, Z backward). R_cv = R_gl * diag(1,-1,-1).
  float c2w[9];
  for (int r = 0; r < 3; r++) for (int col = 0; col < 3; col++) c2w[r * 3 + col] = c.at("rotation")[r][col].get<float>() * (col == 0 ? 1.f : -1.f);
  float pos[3] = {c.at("position")[0].get<float>(), c.at("position")[1].get<float>(), c.at("position")[2].get<float>()};
  for (int r = 0; r < 3; r++) for (int col = 0; col < 3; col++) cam.R[r * 3 + col] = c2w[col * 3 + r];
  for (int r = 0; r < 3; r++) cam.t[r] = -(cam.R[r * 3] * pos[0] + cam.R[r * 3 + 1] * pos[1] + cam.R[r * 3 + 2] * pos[2]);
  cam.update_pos();
  return cam;
}

int render_main(int argc, char** argv) {
  RenderArgs a = parse_render_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(a.device));
  cudaStream_t stream = 0;
  SplatCloud cloud = read_ply(a.splat);
  if (!cloud.has_scales) fail("ply '%s' has no scales", a.splat.c_str());
  log_info("Loaded %zu splats (SH degree %d) from %s", cloud.n, cloud.sh_degree, a.splat.c_str());
  nlohmann::json j; int W = 16, H = 16;   // --export-binding alone renders nothing
  if (!a.cameras.empty()) {
    std::ifstream cf(a.cameras);
    if (!cf) fail("failed to open cameras '%s'", a.cameras.c_str());
    cf >> j;
    W = j.at("width").get<int>(); H = j.at("height").get<int>();
    fs::create_directories(a.output_dir);
  }
  Model model; model.upload(cloud);
  // The active degree the projection kernel evaluates up to (sh_eval stops
  // at `active`); the bands above it stay in the model and are simply not
  // summed, so a degree-3 ply rendered at --sh-degree 0 is its DC colour.
  int sh_degree = a.sh_degree < 0 ? model.degree : std::min(a.sh_degree, model.degree);
  if (a.sh_degree > model.degree) log_warn("--sh-degree %d exceeds the ply's degree %d; rendering at %d", a.sh_degree, model.degree, model.degree);
  if (sh_degree != model.degree) log_info("Rendering with SH bands 0..%d of %d", sh_degree, model.degree);
  RenderCtx ctx; ctx.setup(W, H, model.cap, stream);

  CageRig cage; bool have_cage = false;
  CageApp app; CageOpen open;
  if (!a.cage_app.empty() && a.cage.empty()) fail("--cage-app needs --cage");
  if (!a.cage.empty()) {
    if (!cage.load(a.cage)) fail("--cage %s: cannot open", a.cage.c_str());
    if (!cloud.has_labels) log_warn("--cage: the ply carries no seg_label; every splat binds to the nearest triangle of any layer");
    cage.bind(model, cloud.has_labels ? cloud.labels : std::vector<float>{}, a.cage_min_conf, stream);
    cage.max_growth = a.cage_max_growth;
    cage.fade_start = a.cage_fade_start; cage.fade_end = a.cage_fade_end;
    cage.fill_start = a.cage_fill_start > 0.f ? a.cage_fill_start : a.cage_fade_start;
    cage.fill_end = a.cage_fill_end > 0.f ? a.cage_fill_end : a.cage_fade_end;
    if (cloud.has_fill) {
      cage.set_fill(cloud.fill, cloud.has_gate ? cloud.gate : std::vector<float>{}, stream);
      size_t nfill = 0, ncrease = 0; for (float v : cloud.fill) { nfill += v > 0.f; ncrease += v < 0.f; }
      log_info("Cage fill: %zu filler splats fade in, %zu crease splats fade out over stretch %.2f -> %.2f%s", nfill, ncrease, cage.fill_start, cage.fill_end, cloud.has_gate ? " of their gate vertex pair's distance" : "");
    }
    if (!a.alt_binding.empty()) {
      std::vector<int> as, af; std::vector<float> aw;
      if (!load_alt_binding(a.alt_binding, as, af, aw)) fail("--alt-binding %s: cannot open", a.alt_binding.c_str());
      cage.bind_alt(model, as, af, aw, stream);
      log_info("Dual binding for %d splats", cage.n_alt);
    }
    have_cage = true;
    if (!a.cage_app.empty()) {
      if (!app.load(a.cage_app, cage, stream)) fail("--cage-app %s: cannot open", a.cage_app.c_str());
      log_info("Cage appearance MLP %s", a.cage_app.c_str());
    }
    if (cloud.has_open && !a.cage_open_off && !cage.has_open_theta) log_warn("the ply carries open_* states but the cage no opening gate (B2COPEN1; b2crig: python -m b2crig.rig.open_gate CAGE): rendering the plain splat");
    else if (cloud.has_open && !a.cage_open_off) {
      if (!open.parse_header(read_ply_comments(a.splat))) log_warn("the ply carries open_* states but no b2c.cage_open header comment; gate %g -> %g deg, max d opacity %g assumed", open.start, open.end, open.max_do);
      if (a.cage_open_start > 0.f) open.start = a.cage_open_start;
      if (a.cage_open_end > 0.f) open.end = a.cage_open_end;
      open.init(cage, cloud, stream);
      log_info("Cage open state: gate from the cage (b2crig: partners within %g m, beyond %g m), blend over gate angle %g -> %g deg, max d opacity %g", cage.open_radius, cage.open_min_dist, open.start, open.end, open.max_do);
    } else if (cloud.has_open && a.cage_open_off) log_info("Cage open state ignored (--cage-open-off)");
    if (!a.export_binding.empty()) {
      const int n = model.n;
      auto f = cage.bind_f.download(n, stream); auto b = cage.bind_b.download(n, stream); auto o = cage.bind_off.download(n, stream);
      std::ofstream out(a.export_binding, std::ios::binary);
      if (!out) fail("--export-binding %s: cannot write", a.export_binding.c_str());
      out.write("B2CBIND1", 8); int32_t nn = n; out.write((const char*)&nn, 4);
      out.write((const char*)f.data(), (size_t)n * 4); out.write((const char*)b.data(), (size_t)n * 8); out.write((const char*)o.data(), (size_t)n * 12);
      log_info("Wrote the cage binding of %d splats to %s", n, a.export_binding.c_str());
      if (a.cameras.empty()) return 0;
    }
    if (!a.export_posed.empty()) {
      const int fr = cage.frame_index(a.export_frame);
      if (fr < 0) fail("--export-frame %s: not a frame of %s", a.export_frame.c_str(), a.cage.c_str());
      cage.pose(fr, model, stream);
      auto pv = cage.pos_view.download(model.n, stream), qv = cage.quat_view.download(model.n, stream);
      auto lv = cage.lscale_view.download(model.n, stream), rv = cage.sh_frame.download(model.n, stream);
      SplatCloud c = model.download(stream);
      for (int i = 0; i < model.n; i++) {
        c.pos[(size_t)i * 3] = pv[i].x; c.pos[(size_t)i * 3 + 1] = pv[i].y; c.pos[(size_t)i * 3 + 2] = pv[i].z;
        c.quat[(size_t)i * 4] = qv[i].x; c.quat[(size_t)i * 4 + 1] = qv[i].y; c.quat[(size_t)i * 4 + 2] = qv[i].z; c.quat[(size_t)i * 4 + 3] = qv[i].w;
        c.log_scale[(size_t)i * 3] = lv[i].x; c.log_scale[(size_t)i * 3 + 1] = lv[i].y; c.log_scale[(size_t)i * 3 + 2] = lv[i].z;
      }
      if (cloud.has_labels) { c.labels = cloud.labels; c.has_labels = true; }
      c.has_evidence = false;
      write_ply(a.export_posed, c, {"Exported from Brush", "Vertical axis: y", format("SH degree: %d", c.sh_degree), "SplatRenderMode: default"});
      std::ofstream(a.export_posed + ".frame.f32", std::ios::binary).write((const char*)rv.data(), rv.size() * sizeof(float4));
      log_info("Exported the splat posed by cage frame %s to %s", a.export_frame.c_str(), a.export_posed.c_str());
    }
    log_info("Cage %s: %zu layers, %d vertices, %d triangles, %d frames; %d/%d splats bound across layers", a.cage.c_str(), cage.layers.size(), cage.nv, cage.nf, cage.nframes, cage.n_fallback, model.n);
  }
  const bool want_labels = !a.class_mask.empty() || a.label_maps;
  if (want_labels && !cloud.has_labels) fail("--class-mask / --label-maps need a ply with seg_label");
  // Per-splat class feature buffers: the class mask as one channel, and the label passes three classes at a time.
  DevBuf<float> mask_feat; std::vector<DevBuf<float>> label_feat; int n_classes = 0;
  if (want_labels) {
    for (size_t i = 0; i < cloud.n; i++) n_classes = std::max(n_classes, (int)std::lround(cloud.labels[i * 2]) + 1);
    if (!a.class_mask.empty()) {
      std::vector<float> h(cloud.n * 3, 0.f);
      for (size_t i = 0; i < cloud.n; i++) { int c = (int)std::lround(cloud.labels[i * 2]); for (int m : a.class_mask) if (m == c) h[i * 3] = 1.f; }
      mask_feat.upload(h, stream);
    }
    if (a.label_maps) {
      for (int c0 = 0; c0 < n_classes; c0 += 3) {
        std::vector<float> h(cloud.n * 3, 0.f);
        for (size_t i = 0; i < cloud.n; i++) { int c = (int)std::lround(cloud.labels[i * 2]); if (c >= c0 && c < c0 + 3) h[i * 3 + (c - c0)] = 1.f; }
        label_feat.emplace_back(); label_feat.back().upload(h, stream);
      }
    }
  }

  ConfidenceModel conf;
  bool gated = a.confidence;
  if (gated) {
    if (cloud.has_evidence) {
      log_info("Using the ply's ev_* evidence block");
    } else if (!a.dataset.empty()) {
      Config dc = a.ds_cfg; dc.source = a.dataset; dc.evidence_normal_weight = a.evidence_normal_weight;
      Dataset ds = load_dataset(dc);
      GpuViews gv; gv.upload(ds.train);
      double t0 = now_seconds();
      RenderCtx ectx; ectx.setup(gv.max_w, gv.max_h, model.cap, stream);
      std::vector<int> cviews(ds.train.size(), -1); int cmatched = 0;   // a cage poses the views named like its frames
      if (have_cage)
        for (size_t i = 0; i < ds.train.size(); i++) {
          std::string nm = ds.train[i].name; int fr = cage.frame_index(nm);
          if (fr < 0) { auto d0 = nm.rfind('.'); if (d0 != std::string::npos) fr = cage.frame_index(nm.substr(0, d0)); }
          cviews[i] = fr; cmatched += fr >= 0;
        }
      if (have_cage) log_info("Evidence: %d/%zu dataset views posed by the cage", cmatched, ds.train.size());
      compute_evidence(ectx, model, gv.views, gv.cams, dc, stream, nullptr, nullptr, have_cage ? &cage : nullptr, &cviews);
      cloud.has_evidence = true; cloud.evidence = download_evidence(model, stream);
      log_info("Computed evidence for %zu splats over %zu views in %.1fs", cloud.n, gv.views.size(), now_seconds() - t0);
      if (!a.write_evidence.empty()) {
        std::vector<std::string> cm = {"Exported from Brush", "Vertical axis: y", format("SH degree: %d", cloud.sh_degree), "SplatRenderMode: default"};
        for (auto& c : read_ply_comments(a.splat)) if (c.rfind("b2c.", 0) == 0) cm.push_back(c);   // the subject's header (MHR frame, open-state range) rides along
        write_ply(a.write_evidence, cloud, cm);
        log_info("Wrote evidence to %s", a.write_evidence.c_str());
      }
    } else {
      log_warn("--confidence without evidence: the ply carries no ev_* block and no --dataset was given; every splat is treated as fully trusted");
    }
    if (cloud.has_evidence) conf.build(cloud, a.conf); else conf.build_trusting((int)cloud.n);
  }

  std::vector<unsigned char> img((size_t)W * H * 4), confimg((size_t)W * H);
  double t0 = now_seconds();
  int count = 0;
  for (auto& c : j.at("cameras")) {
    Camera cam = camera_from_json(c, W, H);
    RenderParams p; p.cam = CameraGPU::from(cam, W, H);
    const float* bg = gated ? a.cull : a.background;
    p.bg[0] = bg[0]; p.bg[1] = bg[1]; p.bg[2] = bg[2];
    p.sh_degree = sh_degree; p.bwd_info = false;
    if (gated) { conf.for_camera(cam.pos, stream); p.feat = FeatureMode::Buffer; p.feat_buffer = conf.feature; }
    std::string name = c.at("name").get<std::string>();
    if (have_cage) {
      int fr = cage.frame_index(name);
      if (fr < 0) { auto d0 = name.rfind('.'); if (d0 != std::string::npos) fr = cage.frame_index(name.substr(0, d0)); }
      if (fr < 0) fail("--cage %s has no frame named '%s'", a.cage.c_str(), name.c_str());
      cage.pose(fr, model, stream);
      p.pos_override = cage.pos_view; p.quat_override = cage.quat_view; p.lscale_override = cage.lscale_view; p.sh_frame = cage.sh_frame;
      if (app.on) { app.apply(cage, model, stream); if (a.cage_app_debug) app.debug_alpha(model.n, stream); p.app = app.splat_app; }
      if (open.on) { open.apply(cage, model, stream); if (a.cage_open_debug) open.debug_g(model.n, stream); p.app = open.splat_app; p.app_add = !a.cage_open_debug; }
    }
    render_forward(ctx, model, p, stream);
    auto out = ctx.out_rgba.download((size_t)W * H, stream);
    std::vector<float4> feat;
    if (gated) feat = ctx.out_feat.download((size_t)W * H, stream);
    for (size_t i = 0; i < (size_t)W * H; i++) {
      float4 v = out[i];
      if (gated) {
        float cv = std::min(std::max(feat[i].x, 0.f), 1.f);
        float g = smoothstep_gate(a.gate_lo, a.gate_hi, cv);
        img[i * 4] = to_u8(v.x * g + a.cull[0] * (1.f - g));
        img[i * 4 + 1] = to_u8(v.y * g + a.cull[1] * (1.f - g));
        img[i * 4 + 2] = to_u8(v.z * g + a.cull[2] * (1.f - g));
        img[i * 4 + 3] = to_u8(g);
        confimg[i] = to_u8(cv);
      } else {
        img[i * 4] = to_u8(v.x); img[i * 4 + 1] = to_u8(v.y); img[i * 4 + 2] = to_u8(v.z); img[i * 4 + 3] = to_u8(v.w);
      }
    }
    auto dot = name.rfind('.'); if (dot != std::string::npos) name = name.substr(0, dot);
    fs::path outp = fs::path(a.output_dir) / (name + "." + a.output_format);
    if (!stbi_write_png(outp.string().c_str(), W, H, 4, img.data(), W * 4)) fail("failed to write '%s'", outp.string().c_str());
    if (!a.class_mask.empty()) {
      RenderParams q = p; q.feat = FeatureMode::Buffer; q.feat_buffer = mask_feat;
      render_forward(ctx, model, q, stream);
      auto f = ctx.out_feat.download((size_t)W * H, stream);
      for (size_t i = 0; i < (size_t)W * H; i++) confimg[i] = to_u8(f[i].x);
      fs::path mp = fs::path(a.output_dir) / (name + ".mask." + a.output_format);
      if (!stbi_write_png(mp.string().c_str(), W, H, 1, confimg.data(), W)) fail("failed to write '%s'", mp.string().c_str());
    }
    if (a.label_maps) {
      std::vector<float> best((size_t)W * H, 0.f); std::vector<unsigned char> lab((size_t)W * H, 0);
      for (size_t pass = 0; pass < label_feat.size(); pass++) {
        RenderParams q = p; q.feat = FeatureMode::Buffer; q.feat_buffer = label_feat[pass];
        render_forward(ctx, model, q, stream);
        auto f = ctx.out_feat.download((size_t)W * H, stream);
        for (size_t i = 0; i < (size_t)W * H; i++) {
          const float ch[3] = {f[i].x, f[i].y, f[i].z};
          for (int k = 0; k < 3; k++) { int cl = (int)pass * 3 + k; if (cl < n_classes && ch[k] > best[i]) { best[i] = ch[k]; lab[i] = (unsigned char)cl; } }
        }
      }
      for (size_t i = 0; i < (size_t)W * H; i++) if (out[i].w < 0.5f) lab[i] = 0;
      fs::path lp = fs::path(a.output_dir) / (name + ".labels." + a.output_format);
      if (!stbi_write_png(lp.string().c_str(), W, H, 1, lab.data(), W)) fail("failed to write '%s'", lp.string().c_str());
    }
    if (gated && a.sidecar) {
      fs::path sp = fs::path(a.output_dir) / (name + ".conf." + a.output_format);
      if (!stbi_write_png(sp.string().c_str(), W, H, 1, confimg.data(), W)) fail("failed to write '%s'", sp.string().c_str());
    }
    count++;
  }
  log_info("Rendered %d frames in %.2fs", count, now_seconds() - t0);
  return 0;
}

}  // namespace b2c
