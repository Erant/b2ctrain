#pragma once
#include "dataset/camera.h"
#include "json.hpp"

namespace b2c {
int render_main(int argc, char** argv);
// A body2colmap cameras.json entry (OpenGL camera-to-world rotation, position, fx/fy/cx/cy) as a pinhole Camera at W x H.
Camera camera_from_json(const nlohmann::json& c, int W, int H);
}  // namespace b2c
