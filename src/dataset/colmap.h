#pragma once
#include "dataset/camera.h"
#include <string>
#include <vector>
#include <map>

namespace b2c {

struct ColmapImage {
  int id = 0, camera_id = 0;
  std::string name;
  float q[4] = {1, 0, 0, 0}, t[3] = {0, 0, 0};  // world-to-camera
};
struct ColmapCamera {
  int id = 0; std::string model; int width = 0, height = 0; std::vector<double> params;
  float fx() const;
  float fy() const;
  float cx() const;
  float cy() const;
};
struct Point3D { float x, y, z; unsigned char r, g, b; };

struct ColmapModel {
  std::map<int, ColmapCamera> cameras;
  std::vector<ColmapImage> images;  // sorted by name
  std::vector<Point3D> points;
};

// Reads cameras.txt, images.txt, points3D.txt from `dir`.
ColmapModel read_colmap_text(const std::string& dir);

}  // namespace b2c
