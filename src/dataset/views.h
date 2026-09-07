#pragma once
#include "cli.h"
#include "dataset/camera.h"
#include "dataset/colmap.h"
#include <string>
#include <vector>
#include <cstdint>

namespace b2c {

struct View {
  std::string name;
  Camera cam;              // intrinsics at the loaded resolution
  int w = 0, h = 0;
  bool has_alpha = false;  // frame carries an alpha channel (embedded or from masks/)
  AlphaMode mode = AlphaMode::Transparent;
  std::vector<uint32_t> rgba;     // packed RGBA8 (r | g<<8 | b<<16 | a<<24); premultiplied for transparent views
  std::vector<uint32_t> normals;  // packed RGBA8 normal map, empty if none
  std::vector<uint8_t> weights;   // per-pixel loss weight 0..255, empty if none
  float alpha_coverage = 1.0f;    // mean(a/255), only meaningful for masked views with alpha
  double normal_mask_count = 0;   // sum over pixels of weight where normal alpha > 127
  bool has_normals() const { return !normals.empty(); }
  bool has_weights() const { return !weights.empty(); }
};

struct Dataset {
  std::string root;   // dataset directory
  std::string name;   // folder name (for {dataset} interpolation)
  std::vector<View> train, eval;
  std::vector<Point3D> points;
  std::string init_ply;  // path of init ply if present, else empty
  int n_masked = 0, n_transparent = 0, n_weighted = 0, n_normals = 0;
};

Dataset load_dataset(const Config& cfg);

}  // namespace b2c
