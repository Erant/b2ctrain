#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace b2c {

// A triangle mesh in the dataset's world frame (the body proxy the hollow loss measures depth against).
struct TriMesh {
  std::vector<float> v;      // [nv][3]
  std::vector<uint32_t> f;   // [nf][3]
  size_t nv() const { return v.size() / 3; }
  size_t nf() const { return f.size() / 3; }
};

// Reads a triangle mesh from a .ply (ascii or binary little-endian; vertex x/y/z as float or double, faces as a
// vertex_indices list; polygons are fan-triangulated) or a Wavefront .obj (v / f lines).
TriMesh read_mesh(const std::string& path);

}  // namespace b2c
