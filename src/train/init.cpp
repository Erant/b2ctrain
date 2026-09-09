#include "train/init.h"
#include "util/log.h"
#include <algorithm>
#include <cmath>
#include <thread>
#include <atomic>

namespace b2c {

float Bounds::median_size() const {
  float e[3] = {extent[0], extent[1], extent[2]};
  std::sort(e, e + 3);
  return e[1];
}
float Bounds::max_extent() const { return std::max(extent[0], std::max(extent[1], extent[2])); }

Bounds bounds_from_pos(const float* pos, size_t n, float percentile) {
  Bounds b{};
  for (int a = 0; a < 3; a++) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; i++) v[i] = pos[i * 3 + a];
    std::sort(v.begin(), v.end());
    size_t lo = (size_t)((1.f - percentile) / 2.f * n);
    size_t hi = std::min(n - 1, (size_t)((1.f + percentile) / 2.f * n));
    b.min[a] = v[lo]; b.max[a] = v[hi];
    b.center[a] = 0.5f * (b.min[a] + b.max[a]);
    b.extent[a] = b.max[a] - b.min[a];
  }
  return b;
}

float median_nn_distance(const float* pos, size_t n, size_t samples) {
  if (n < 2) return 0.f;
  size_t step = std::max<size_t>(1, n / std::max<size_t>(samples, 1));
  std::vector<float> d;
  for (size_t p = 0; p < n; p += step) {
    float px = pos[p * 3], py = pos[p * 3 + 1], pz = pos[p * 3 + 2], best = INFINITY;
    for (size_t q = 0; q < n; q++) {
      if (q == p) continue;
      float dx = pos[q * 3] - px, dy = pos[q * 3 + 1] - py, dz = pos[q * 3 + 2] - pz;
      best = std::min(best, dx * dx + dy * dy + dz * dz);
    }
    if (std::isfinite(best)) d.push_back(std::sqrt(best));
  }
  if (d.empty()) return 0.f;
  std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
  return d[d.size() / 2];
}

namespace {
// Scales from the 2nd and 3rd nearest neighbour distances: ln(clamp((d1 + d2) / 4, 1e-3, 0.1 * median_size)).
std::vector<float> knn_log_scales(const std::vector<float>& pos) {
  size_t n = pos.size() / 3;
  std::vector<float> out(n * 3, 0.f);
  if (n < 3) return out;
  Bounds b = bounds_from_pos(pos.data(), n, 0.75f);
  float median_size = std::max(b.median_size(), 0.01f);
  float cap = median_size * 0.1f;
  if (n > 200000) log_warn("kNN scale init on %zu points is O(n^2) and will be slow", n);
  unsigned nt = std::max(1u, std::min(std::thread::hardware_concurrency(), 32u));
  std::atomic<size_t> next{0};
  auto worker = [&]() {
    for (size_t i; (i = next.fetch_add(256)) < n;) {
      size_t end = std::min(n, i + 256);
      for (size_t p = i; p < end; p++) {
        float px = pos[p * 3], py = pos[p * 3 + 1], pz = pos[p * 3 + 2];
        float d1 = INFINITY, d2 = INFINITY;
        for (size_t q = 0; q < n; q++) {
          if (q == p) continue;
          float dx = pos[q * 3] - px, dy = pos[q * 3 + 1] - py, dz = pos[q * 3 + 2] - pz;
          float d = dx * dx + dy * dy + dz * dz;
          if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) d2 = d;
        }
        float dist = (std::sqrt(d1) + std::sqrt(d2)) / 4.f;
        float ls = std::log(std::min(std::max(dist, 1e-3f), cap));
        out[p * 3] = out[p * 3 + 1] = out[p * 3 + 2] = ls;
      }
    }
  };
  std::vector<std::thread> th;
  for (unsigned t = 0; t < nt; t++) th.emplace_back(worker);
  for (auto& t : th) t.join();
  return out;
}
}  // namespace

SplatCloud initial_splats(const Dataset& ds, const Config& cfg) {
  SplatCloud c;
  if (!ds.init_ply.empty()) {
    c = read_ply(ds.init_ply);
    log_info("Loaded %zu splats from '%s' (SH degree %d)", c.n, ds.init_ply.c_str(), c.sh_degree);
  } else {
    std::vector<Point3D> pts = ds.points;
    if (cfg.subsample_points && *cfg.subsample_points > 1) {
      std::vector<Point3D> sub;
      for (size_t i = 0; i < pts.size(); i += *cfg.subsample_points) sub.push_back(pts[i]);
      pts = sub;
    }
    if (pts.empty()) fail("no initial points (points3D.txt is empty and no init.ply was found); random init is not supported");
    c.has_scales = false;
    c.resize(pts.size(), 0);
    for (size_t i = 0; i < pts.size(); i++) {
      c.pos[i * 3] = pts[i].x; c.pos[i * 3 + 1] = pts[i].y; c.pos[i * 3 + 2] = pts[i].z;
      c.sh[i * 3] = (pts[i].r / 255.f - 0.5f) / SH_C0; c.sh[i * 3 + 1] = (pts[i].g / 255.f - 0.5f) / SH_C0; c.sh[i * 3 + 2] = (pts[i].b / 255.f - 0.5f) / SH_C0;
      c.opacity[i] = 0.f;  // logit(0.5)
    }
  }
  // Cap to max_splats (strided).
  if (c.n > cfg.max_splats) {
    size_t step = (c.n + cfg.max_splats - 1) / cfg.max_splats;
    log_warn("initial cloud has %zu points, subsampling by %zu to respect --max-splats", c.n, step);
    SplatCloud s; s.has_scales = c.has_scales; s.has_evidence = false;
    size_t m = (c.n + step - 1) / step; s.resize(m, c.sh_degree);
    int K = c.K();
    for (size_t j = 0, i = 0; i < c.n; i += step, j++) {
      for (int k = 0; k < 3; k++) { s.pos[j * 3 + k] = c.pos[i * 3 + k]; s.log_scale[j * 3 + k] = c.log_scale[i * 3 + k]; }
      for (int k = 0; k < 4; k++) s.quat[j * 4 + k] = c.quat[i * 4 + k];
      s.opacity[j] = c.opacity[i];
      for (int k = 0; k < K * 3; k++) s.sh[j * K * 3 + k] = c.sh[i * K * 3 + k];
    }
    c = std::move(s);
  }
  if (!c.has_scales) { c.log_scale = knn_log_scales(c.pos); c.has_scales = true; }
  // Pad / truncate SH to the configured degree.
  int Kd = sh_coeffs_for_degree((int)cfg.sh_degree);
  if (c.K() != Kd) {
    int Ko = c.K();
    std::vector<float> sh((size_t)c.n * Kd * 3, 0.f);
    for (size_t i = 0; i < c.n; i++)
      for (int k = 0; k < std::min(Ko, Kd); k++)
        for (int ch = 0; ch < 3; ch++) sh[(i * Kd + k) * 3 + ch] = c.sh[(i * Ko + k) * 3 + ch];
    c.sh = std::move(sh); c.sh_degree = (int)cfg.sh_degree;
  }
  c.has_evidence = false; c.evidence.clear();
  return c;
}

}  // namespace b2c
