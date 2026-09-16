#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace b2c {

// A triangle mesh in the dataset's world frame (the body proxy the hollow loss measures depth against, and the
// meshification chain's meshes). Colours and normals are optional and empty when the file has none; UVs come from
// an OBJ with `vt` and `f v/vt` records (one UV per corner: fuv indexes uv, parallel to f).
struct TriMesh {
  std::vector<float> v;      // [nv][3]
  std::vector<uint32_t> f;   // [nf][3]
  std::vector<uint8_t> col;  // [nv][3] rgb, or empty
  std::vector<float> nrm;    // [nv][3], or empty
  std::vector<float> uv;     // [nuv][2], or empty
  std::vector<uint32_t> fuv; // [nf][3] into uv, or empty
  size_t nv() const { return v.size() / 3; }
  size_t nf() const { return f.size() / 3; }
  bool has_colour() const { return col.size() == v.size(); }
  bool has_normals() const { return nrm.size() == v.size(); }
  bool has_uv() const { return !uv.empty() && fuv.size() == f.size(); }
};

// Reads a triangle mesh from a .ply (ascii or binary little-endian; vertex x/y/z as float or double, red/green/blue and
// nx/ny/nz when present, faces as a vertex_indices list; polygons are fan-triangulated) or a Wavefront .obj (v / vt / f lines).
TriMesh read_mesh(const std::string& path);
// Binary little-endian PLY: float x/y/z [nx/ny/nz] [uchar red/green/blue], faces as `list uchar uint vertex_indices`.
void write_ply_mesh(const std::string& path, const TriMesh& m);
// Wavefront OBJ with per-corner UVs (`vt` rows, `f v/vt` faces, `mtllib` + `usemtl tex`) and its .mtl next to it
// naming `texture` as map_Kd. v = 1 is the texture's top row (the standard convention).
void write_obj_mesh(const std::string& path, const TriMesh& m, const std::string& texture);

}  // namespace b2c
