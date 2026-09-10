#include "dataset/views.h"
#include "util/log.h"
#include "ply.h"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
#include <filesystem>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>
#include <cctype>

namespace b2c {
namespace fs = std::filesystem;

namespace {
std::string lower(std::string s) { for (auto& c : s) c = (char)std::tolower((unsigned char)c); return s; }
std::string file_stem(const std::string& name) { auto p = name.rfind('.'); return p == std::string::npos ? name : name.substr(0, p); }
bool is_sidecar_dir(const std::string& comp) { return comp == "masks" || comp == "normals" || comp == "weights"; }

// Files under <root>/<dir>/ (recursively), keyed by lowercase stem and lowercase filename.
struct SidecarIndex {
  std::vector<std::pair<std::string, std::string>> entries;  // (lowercase file name, path)
  void build(const fs::path& dir) {
    if (!fs::is_directory(dir)) return;
    for (auto& e : fs::recursive_directory_iterator(dir))
      if (e.is_regular_file()) entries.emplace_back(lower(e.path().filename().string()), e.path().string());
  }
  // brush: candidate's stem must equal the image's file name, its stem, or (masks) stem + ".mask".
  std::string find(const std::string& image_name, bool allow_mask_suffix) const {
    std::string n = lower(image_name), s = lower(file_stem(image_name));
    for (auto& [fname, path] : entries) {
      std::string fstem = file_stem(fname);
      if (fstem == n || fstem == s || (allow_mask_suffix && fstem == s + ".mask")) return path;
    }
    return "";
  }
};

std::string find_image(const fs::path& root, const std::string& name) {
  fs::path direct = root / "images" / name;
  if (fs::exists(direct)) return direct.string();
  direct = root / name;
  if (fs::exists(direct)) return direct.string();
  std::string want = "/" + name;
  for (auto& e : fs::recursive_directory_iterator(root)) {
    if (!e.is_regular_file()) continue;
    std::string p = e.path().string();
    if (p.size() >= want.size() && p.compare(p.size() - want.size(), want.size(), want) == 0) {
      bool sidecar = false;
      for (auto& comp : fs::relative(e.path(), root)) if (is_sidecar_dir(comp.string())) sidecar = true;
      if (!sidecar) return p;
    }
  }
  return "";
}

struct Decoded { int w = 0, h = 0, channels_in_file = 0; std::vector<unsigned char> rgba; };
Decoded decode_rgba(const std::string& path) {
  Decoded d;
  int c = 0;
  unsigned char* p = stbi_load(path.c_str(), &d.w, &d.h, &c, 4);
  if (!p) fail("failed to decode '%s': %s", path.c_str(), stbi_failure_reason());
  d.channels_in_file = c;
  d.rgba.assign(p, p + (size_t)d.w * d.h * 4);
  stbi_image_free(p);
  return d;
}
// Resize a 4-channel 8-bit image treating channels independently (no alpha weighting), Mitchell filter.
std::vector<unsigned char> resize_rgba(const std::vector<unsigned char>& src, int w, int h, int nw, int nh, stbir_filter filter) {
  std::vector<unsigned char> out((size_t)nw * nh * 4);
  stbir_resize(src.data(), w, h, 0, out.data(), nw, nh, 0, STBIR_4CHANNEL, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, filter);
  return out;
}
}  // namespace

Dataset load_dataset(const Config& cfg) {
  Dataset ds;
  fs::path root = fs::absolute(cfg.source);
  if (!fs::is_directory(root)) fail("dataset path '%s' is not a directory", root.string().c_str());
  ds.root = root.string();
  ds.name = root.filename().string();
  if (ds.name.empty()) ds.name = root.parent_path().filename().string();

  // Find the COLMAP text model: cameras.txt at root or in a subdirectory (prefer root, then sparse/0).
  fs::path model_dir;
  for (auto cand : {root, root / "sparse" / "0", root / "sparse", root / "colmap" / "sparse" / "0"})
    if (fs::exists(cand / "cameras.txt")) { model_dir = cand; break; }
  if (model_dir.empty()) {
    for (auto& e : fs::recursive_directory_iterator(root))
      if (e.is_regular_file() && e.path().filename() == "cameras.txt") { model_dir = e.path().parent_path(); break; }
  }
  if (model_dir.empty()) fail("no cameras.txt found under '%s' (only COLMAP text datasets are supported)", root.string().c_str());
  ColmapModel model = read_colmap_text(model_dir.string());
  ds.points = model.points;

  // init.ply preferred, else the lexicographically last .ply in the dataset. Triangle meshes are
  // skipped: the hollow loss takes its proxy from <dataset>/mesh.ply, which is not a splat cloud.
  {
    std::vector<std::string> plys;
    for (auto& e : fs::recursive_directory_iterator(root, fs::directory_options::follow_directory_symlink))
      if (e.is_regular_file() && lower(e.path().extension().string()) == ".ply" && ply_is_vertex_only(e.path().string()))
        plys.push_back(e.path().string());
    std::sort(plys.begin(), plys.end());
    for (auto& p : plys) if (fs::path(p).filename() == "init.ply") ds.init_ply = p;
    if (ds.init_ply.empty() && !plys.empty()) ds.init_ply = plys.back();
  }

  // Frame list: sorted by name, subsampled, capped.
  std::vector<ColmapImage> images = model.images;
  if (cfg.subsample_frames && *cfg.subsample_frames > 1) {
    std::vector<ColmapImage> sub;
    for (size_t i = 0; i < images.size(); i += *cfg.subsample_frames) sub.push_back(images[i]);
    images = sub;
  }
  if (cfg.max_frames && images.size() > *cfg.max_frames) images.resize(*cfg.max_frames);

  SidecarIndex masks, normals, weights;
  masks.build(root / "masks"); normals.build(root / "normals"); weights.build(root / "weights");
  if (masks.entries.empty() || normals.entries.empty() || weights.entries.empty()) {
    // Sidecar dirs may live next to the model dir instead.
    if (model_dir != root) { if (masks.entries.empty()) masks.build(model_dir / "masks"); if (normals.entries.empty()) normals.build(model_dir / "normals"); if (weights.entries.empty()) weights.build(model_dir / "weights"); }
  }

  struct Job { ColmapImage im; std::string img_path, mask_path, normal_path, weight_path; };
  std::vector<Job> jobs;
  for (auto& im : images) {
    Job j; j.im = im;
    j.img_path = find_image(root, im.name);
    if (j.img_path.empty()) { log_warn("Skipped '%s': image file not found", im.name.c_str()); continue; }
    j.mask_path = masks.find(im.name, true);
    j.normal_path = normals.find(im.name, false);
    j.weight_path = weights.find(im.name, false);
    jobs.push_back(j);
  }
  if (jobs.empty()) fail("dataset has no usable views");

  std::vector<View> views(jobs.size());
  std::atomic<size_t> next{0};
  std::mutex warn_mutex;
  std::vector<std::string> warnings;
  auto worker = [&]() {
    for (size_t i; (i = next.fetch_add(1)) < jobs.size();) {
      const Job& j = jobs[i];
      View& v = views[i];
      v.name = j.im.name;
      auto cam_it = model.cameras.find(j.im.camera_id);
      if (cam_it == model.cameras.end()) fail("images.txt: unknown camera id %d for '%s'", j.im.camera_id, j.im.name.c_str());
      const ColmapCamera& cc = cam_it->second;
      Decoded d = decode_rgba(j.img_path);
      bool embedded_alpha = d.channels_in_file == 4 || d.channels_in_file == 2;
      // Mask sidecar: resized (triangle) to the image size, alpha from its alpha (if any) else red channel.
      if (!j.mask_path.empty()) {
        Decoded m = decode_rgba(j.mask_path);
        std::vector<unsigned char> mr = (m.w == d.w && m.h == d.h) ? m.rgba : resize_rgba(m.rgba, m.w, m.h, d.w, d.h, STBIR_FILTER_TRIANGLE);
        bool mask_has_alpha = m.channels_in_file == 4 || m.channels_in_file == 2;
        for (size_t p = 0; p < (size_t)d.w * d.h; p++) d.rgba[p * 4 + 3] = mask_has_alpha ? mr[p * 4 + 3] : mr[p * 4];
      }
      v.has_alpha = embedded_alpha || !j.mask_path.empty();
      v.mode = cfg.alpha_mode ? *cfg.alpha_mode : (!j.mask_path.empty() ? AlphaMode::Masked : AlphaMode::Transparent);
      // Resolution cap on the long edge.
      float scale = 1.0f;
      int longest = std::max(d.w, d.h);
      if ((int)cfg.max_resolution < longest) scale = (float)cfg.max_resolution / (float)longest;
      int nw = std::max(1, (int)std::lround(d.w * scale)), nh = std::max(1, (int)std::lround(d.h * scale));
      if (nw != d.w || nh != d.h) { d.rgba = resize_rgba(d.rgba, d.w, d.h, nw, nh, STBIR_FILTER_MITCHELL); }
      v.w = nw; v.h = nh;
      v.cam.width = cc.width; v.cam.height = cc.height;
      v.cam.fx = cc.fx(); v.cam.fy = cc.fy(); v.cam.cx = cc.cx(); v.cam.cy = cc.cy();
      v.cam.rescale(nw, nh);
      v.cam.set_w2c_quat(j.im.q[0], j.im.q[1], j.im.q[2], j.im.q[3], j.im.t[0], j.im.t[1], j.im.t[2]);
      if (!std::isfinite(v.cam.pos[0] + v.cam.pos[1] + v.cam.pos[2])) fail("camera for '%s' contains nan or inf", v.name.c_str());
      // Pack. Transparent GT is premultiplied in byte space like brush.
      v.rgba.resize((size_t)nw * nh);
      double alpha_sum = 0;
      for (size_t p = 0; p < (size_t)nw * nh; p++) {
        unsigned r = d.rgba[p * 4], g = d.rgba[p * 4 + 1], b = d.rgba[p * 4 + 2], a = v.has_alpha ? d.rgba[p * 4 + 3] : 255;
        if (v.mode == AlphaMode::Transparent && v.has_alpha) { r = (r * a + 127) / 255; g = (g * a + 127) / 255; b = (b * a + 127) / 255; }
        v.rgba[p] = r | (g << 8) | (b << 16) | (a << 24);
        alpha_sum += a;
      }
      v.alpha_coverage = (float)(alpha_sum / (255.0 * nw * nh));
      if (!j.weight_path.empty()) {
        Decoded wm = decode_rgba(j.weight_path);
        // Luma of RGB, own alpha ignored, triangle resize.
        std::vector<unsigned char> wr = (wm.w == nw && wm.h == nh) ? wm.rgba : resize_rgba(wm.rgba, wm.w, wm.h, nw, nh, STBIR_FILTER_TRIANGLE);
        v.weights.resize((size_t)nw * nh);
        for (size_t p = 0; p < (size_t)nw * nh; p++) {
          if (wm.channels_in_file >= 3) v.weights[p] = (unsigned char)std::lround(0.2126 * wr[p * 4] + 0.7152 * wr[p * 4 + 1] + 0.0722 * wr[p * 4 + 2]);
          else v.weights[p] = wr[p * 4];
        }
      }
      if (!j.normal_path.empty()) {
        Decoded nm = decode_rgba(j.normal_path);
        std::vector<unsigned char> nr = (nm.w == nw && nm.h == nh) ? nm.rgba : resize_rgba(nm.rgba, nm.w, nm.h, nw, nh, STBIR_FILTER_POINT_SAMPLE);
        bool n_alpha = nm.channels_in_file == 4 || nm.channels_in_file == 2;
        v.normals.resize((size_t)nw * nh);
        double cnt = 0;
        for (size_t p = 0; p < (size_t)nw * nh; p++) {
          unsigned a = n_alpha ? nr[p * 4 + 3] : 255;
          v.normals[p] = nr[p * 4] | (nr[p * 4 + 1] << 8) | (nr[p * 4 + 2] << 16) | (a << 24);
          if (a > 127) cnt += v.has_weights() ? v.weights[p] / 255.0 : 1.0;
        }
        v.normal_mask_count = cnt;
      }
    }
  };
  unsigned nt = std::max(1u, std::min(std::thread::hardware_concurrency(), 16u));
  std::vector<std::thread> threads;
  std::exception_ptr err;
  std::mutex err_mutex;
  for (unsigned t = 0; t < nt; t++) threads.emplace_back([&]() { try { worker(); } catch (...) { std::lock_guard<std::mutex> l(err_mutex); if (!err) err = std::current_exception(); } });
  for (auto& t : threads) t.join();
  if (err) std::rethrow_exception(err);

  // Eval split: view index i goes to eval when i % every == 0.
  for (size_t i = 0; i < views.size(); i++) {
    View& v = views[i];
    if (v.has_alpha) { if (v.mode == AlphaMode::Masked) ds.n_masked++; else ds.n_transparent++; }
    if (v.has_weights()) ds.n_weighted++;
    if (v.has_normals()) ds.n_normals++;
    if (cfg.eval_split_every && *cfg.eval_split_every > 0 && i % *cfg.eval_split_every == 0) ds.eval.push_back(std::move(v));
    else ds.train.push_back(std::move(v));
  }
  if (ds.train.empty()) fail("dataset has no training views");
  log_info("Loaded dataset with %zu training, %zu eval views", ds.train.size(), ds.eval.size());
  log_info("Dataset alpha modes: %d masked, %d transparent view(s)", ds.n_masked, ds.n_transparent);
  if (cfg.alpha_mode) {
    int with_sidecar = 0; for (auto& j : jobs) if (!j.mask_path.empty()) with_sidecar++;
    if (with_sidecar > 0 && with_sidecar < (int)jobs.size())
      log_warn("--alpha-mode forces every view: %d view(s) have a masks/ sidecar and %d do not", with_sidecar, (int)jobs.size() - with_sidecar);
  }
  if (ds.n_weighted) log_info("Dataset loss weights: %d view(s) carry a weights/ sidecar", ds.n_weighted);
  if (ds.n_normals) log_info("Dataset normals: %d view(s) carry a normals/ sidecar", ds.n_normals);
  if (!ds.init_ply.empty()) log_info("Using '%s' as the initial splat", ds.init_ply.c_str());
  else log_info("Initial point cloud: %zu points", ds.points.size());
  return ds;
}

}  // namespace b2c
