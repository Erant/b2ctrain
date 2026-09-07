#include "dataset/colmap.h"
#include "util/log.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>

namespace b2c {
namespace fs = std::filesystem;

float ColmapCamera::fx() const { return (float)params[0]; }
float ColmapCamera::fy() const { return (float)(model == "SIMPLE_PINHOLE" || model == "SIMPLE_RADIAL" || model == "RADIAL" ? params[0] : params[1]); }
float ColmapCamera::cx() const { return (float)(model == "SIMPLE_PINHOLE" || model == "SIMPLE_RADIAL" || model == "RADIAL" ? params[1] : params[2]); }
float ColmapCamera::cy() const { return (float)(model == "SIMPLE_PINHOLE" || model == "SIMPLE_RADIAL" || model == "RADIAL" ? params[2] : params[3]); }

static std::ifstream open_or_fail(const fs::path& p) {
  std::ifstream f(p);
  if (!f) fail("failed to open '%s'", p.string().c_str());
  return f;
}

ColmapModel read_colmap_text(const std::string& dir) {
  ColmapModel m;
  {
    auto f = open_or_fail(fs::path(dir) / "cameras.txt");
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream ss(line);
      ColmapCamera c; ss >> c.id >> c.model >> c.width >> c.height;
      double v; while (ss >> v) c.params.push_back(v);
      if (c.model == "PINHOLE") { if (c.params.size() < 4) fail("cameras.txt: PINHOLE needs 4 params"); }
      else if (c.model == "SIMPLE_PINHOLE") { if (c.params.size() < 3) fail("cameras.txt: SIMPLE_PINHOLE needs 3 params"); }
      else if (c.model == "SIMPLE_RADIAL" || c.model == "RADIAL" || c.model == "OPENCV") {
        log_warn("camera model %s: distortion is ignored (pinhole intrinsics only)", c.model.c_str());
      } else fail("unsupported camera model '%s' (only pinhole models are supported)", c.model.c_str());
      m.cameras[c.id] = c;
    }
  }
  {
    auto f = open_or_fail(fs::path(dir) / "images.txt");
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream ss(line);
      ColmapImage im;
      ss >> im.id >> im.q[0] >> im.q[1] >> im.q[2] >> im.q[3] >> im.t[0] >> im.t[1] >> im.t[2] >> im.camera_id >> im.name;
      if (im.name.empty()) fail("images.txt: malformed line '%s'", line.c_str());
      m.images.push_back(im);
      // The POINTS2D line (possibly empty) follows.
      std::getline(f, line);
    }
    std::sort(m.images.begin(), m.images.end(), [](auto& a, auto& b) { return a.name < b.name; });
  }
  {
    fs::path p = fs::path(dir) / "points3D.txt";
    if (fs::exists(p)) {
      auto f = open_or_fail(p);
      std::string line;
      while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        long id; Point3D pt; int r, g, b;
        ss >> id >> pt.x >> pt.y >> pt.z >> r >> g >> b;
        pt.r = (unsigned char)r; pt.g = (unsigned char)g; pt.b = (unsigned char)b;
        m.points.push_back(pt);
      }
    }
  }
  return m;
}

}  // namespace b2c
