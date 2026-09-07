#include "render/renderer.h"
#include "gpu/render.h"
#include "ply.h"
#include "util/log.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "json.hpp"
#include <fstream>
#include <filesystem>
#include <cstring>

namespace b2c {
namespace fs = std::filesystem;

namespace {
struct RenderArgs {
  std::string splat, cameras, output_dir = "out", output_format = "png";
  float background[3] = {1.f, 1.f, 1.f};
  bool confidence = false;
  int device = 0;
};

RenderArgs parse_render_args(int argc, char** argv) {
  RenderArgs a;
  for (int i = 2; i < argc; i++) {
    std::string k = argv[i];
    auto val = [&](const char* name) -> std::string { if (i + 1 >= argc) fail("a value is required for '%s'", name); return argv[++i]; };
    if (k == "--splat") a.splat = val("--splat");
    else if (k == "--cameras") a.cameras = val("--cameras");
    else if (k == "--output-dir") a.output_dir = val("--output-dir");
    else if (k == "--output-format") a.output_format = val("--output-format");
    else if (k == "--background") { auto v = val("--background"); if (sscanf(v.c_str(), "%f,%f,%f", &a.background[0], &a.background[1], &a.background[2]) != 3) fail("invalid --background '%s'", v.c_str()); }
    else if (k == "--device") a.device = atoi(val("--device").c_str());
    else if (k == "--confidence") a.confidence = true;
    else if (k == "-h" || k == "--help") { printf("Usage: b2ctrain render --splat <ply> --cameras <cameras.json> [--output-dir out] [--background r,g,b] [--output-format png]\n"); exit(0); }
    else fail("unexpected argument '%s' found", k.c_str());
  }
  if (a.splat.empty() || a.cameras.empty()) fail("--splat and --cameras are required");
  if (a.output_format != "png") fail("only png output is supported");
  if (a.confidence) fail("--confidence is not implemented yet");
  return a;
}
}  // namespace

int render_main(int argc, char** argv) {
  RenderArgs a = parse_render_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(a.device));
  SplatCloud cloud = read_ply(a.splat);
  if (!cloud.has_scales) fail("ply '%s' has no scales", a.splat.c_str());
  log_info("Loaded %zu splats (SH degree %d) from %s", cloud.n, cloud.sh_degree, a.splat.c_str());
  std::ifstream cf(a.cameras);
  if (!cf) fail("failed to open cameras '%s'", a.cameras.c_str());
  nlohmann::json j; cf >> j;
  int W = j.at("width").get<int>(), H = j.at("height").get<int>();
  fs::create_directories(a.output_dir);
  Model model; model.upload(cloud);
  RenderCtx ctx; ctx.setup(W, H, model.cap, 0);
  std::vector<unsigned char> img((size_t)W * H * 4);
  double t0 = now_seconds();
  int count = 0;
  for (auto& c : j.at("cameras")) {
    Camera cam;
    cam.width = W; cam.height = H;
    cam.fx = c.at("fx").get<float>(); cam.fy = c.at("fy").get<float>(); cam.cx = c.at("cx").get<float>(); cam.cy = c.at("cy").get<float>();
    // rotation: OpenGL camera-to-world (columns = local axes, Y up, Z backward). Convert: R_cv_c2w = R_gl * diag(1,-1,-1).
    float c2w[9];
    for (int r = 0; r < 3; r++) for (int col = 0; col < 3; col++) c2w[r * 3 + col] = c.at("rotation")[r][col].get<float>() * (col == 0 ? 1.f : -1.f);
    float pos[3] = {c.at("position")[0].get<float>(), c.at("position")[1].get<float>(), c.at("position")[2].get<float>()};
    // w2c: R = c2w^T, t = -R pos
    for (int r = 0; r < 3; r++) for (int col = 0; col < 3; col++) cam.R[r * 3 + col] = c2w[col * 3 + r];
    for (int r = 0; r < 3; r++) cam.t[r] = -(cam.R[r * 3] * pos[0] + cam.R[r * 3 + 1] * pos[1] + cam.R[r * 3 + 2] * pos[2]);
    cam.update_pos();
    RenderParams p; p.cam = CameraGPU::from(cam, W, H);
    p.bg[0] = a.background[0]; p.bg[1] = a.background[1]; p.bg[2] = a.background[2];
    p.sh_degree = model.degree; p.bwd_info = false;
    render_forward(ctx, model, p, 0);
    auto out = ctx.out_rgba.download((size_t)W * H);
    for (size_t i = 0; i < (size_t)W * H; i++) {
      float4 v = out[i];
      img[i * 4] = (unsigned char)fminf(fmaxf(v.x * 255.f + 0.5f, 0.f), 255.f);
      img[i * 4 + 1] = (unsigned char)fminf(fmaxf(v.y * 255.f + 0.5f, 0.f), 255.f);
      img[i * 4 + 2] = (unsigned char)fminf(fmaxf(v.z * 255.f + 0.5f, 0.f), 255.f);
      img[i * 4 + 3] = (unsigned char)fminf(fmaxf(v.w * 255.f + 0.5f, 0.f), 255.f);
    }
    std::string name = c.at("name").get<std::string>();
    auto dot = name.rfind('.'); if (dot != std::string::npos) name = name.substr(0, dot);
    fs::path outp = fs::path(a.output_dir) / (name + "." + a.output_format);
    if (!stbi_write_png(outp.string().c_str(), W, H, 4, img.data(), W * 4)) fail("failed to write '%s'", outp.string().c_str());
    count++;
  }
  log_info("Rendered %d frames in %.2fs", count, now_seconds() - t0);
  return 0;
}

}  // namespace b2c
