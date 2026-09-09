#include "render/renderer.h"
#include "render/confidence.h"
#include "gpu/render.h"
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
"      --device <N>                     CUDA device [default: 0]\n\n"
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
  if (a.splat.empty() || a.cameras.empty()) fail("the following required arguments were not provided: --splat <SPLAT> --cameras <CAMERAS>");
  if (a.output_format != "png") fail("only png output is supported");
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
  std::ifstream cf(a.cameras);
  if (!cf) fail("failed to open cameras '%s'", a.cameras.c_str());
  nlohmann::json j; cf >> j;
  int W = j.at("width").get<int>(), H = j.at("height").get<int>();
  fs::create_directories(a.output_dir);
  Model model; model.upload(cloud);
  RenderCtx ctx; ctx.setup(W, H, model.cap, stream);

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
      compute_evidence(ectx, model, gv.views, gv.cams, dc, stream);
      cloud.has_evidence = true; cloud.evidence = download_evidence(model, stream);
      log_info("Computed evidence for %zu splats over %zu views in %.1fs", cloud.n, gv.views.size(), now_seconds() - t0);
      if (!a.write_evidence.empty()) {
        write_ply(a.write_evidence, cloud, {"Exported from Brush", "Vertical axis: y", format("SH degree: %d", cloud.sh_degree), "SplatRenderMode: default"});
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
    p.sh_degree = model.degree; p.bwd_info = false;
    if (gated) { conf.for_camera(cam.pos, stream); p.feat = FeatureMode::Buffer; p.feat_buffer = conf.feature; }
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
    std::string name = c.at("name").get<std::string>();
    auto dot = name.rfind('.'); if (dot != std::string::npos) name = name.substr(0, dot);
    fs::path outp = fs::path(a.output_dir) / (name + "." + a.output_format);
    if (!stbi_write_png(outp.string().c_str(), W, H, 4, img.data(), W * 4)) fail("failed to write '%s'", outp.string().c_str());
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
