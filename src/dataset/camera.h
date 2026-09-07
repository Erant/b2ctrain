#pragma once
#include <string>
#include <cmath>

namespace b2c {

// Pinhole camera. R/t are world-to-camera (OpenCV: +X right, +Y down, +Z forward), R row-major.
struct Camera {
  int width = 0, height = 0;
  float fx = 0, fy = 0, cx = 0, cy = 0;
  float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  float t[3] = {0, 0, 0};
  float pos[3] = {0, 0, 0};  // camera center in world space

  void set_w2c_quat(float qw, float qx, float qy, float qz, float tx, float ty, float tz) {
    float n = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    qw /= n; qx /= n; qy /= n; qz /= n;
    R[0] = 1 - 2 * (qy * qy + qz * qz); R[1] = 2 * (qx * qy - qz * qw); R[2] = 2 * (qx * qz + qy * qw);
    R[3] = 2 * (qx * qy + qz * qw);     R[4] = 1 - 2 * (qx * qx + qz * qz); R[5] = 2 * (qy * qz - qx * qw);
    R[6] = 2 * (qx * qz - qy * qw);     R[7] = 2 * (qy * qz + qx * qw);     R[8] = 1 - 2 * (qx * qx + qy * qy);
    t[0] = tx; t[1] = ty; t[2] = tz;
    update_pos();
  }
  void update_pos() {
    // pos = -R^T t
    pos[0] = -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]);
    pos[1] = -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]);
    pos[2] = -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]);
  }
  // Rescale intrinsics to a new image size (uniform scale assumed).
  void rescale(int new_w, int new_h) {
    float sx = (float)new_w / (float)width, sy = (float)new_h / (float)height;
    fx *= sx; cx *= sx; fy *= sy; cy *= sy; width = new_w; height = new_h;
  }
  float focal_px() const { return 0.5f * (fx + fy); }
};

}  // namespace b2c
