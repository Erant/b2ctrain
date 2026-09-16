#pragma once
// Shared pieces of the meshification subcommands (`b2ctrain mesh-*`): the cameras.json set, image files, raw float
// maps, a small option parser, and a parallel-for for the PNG decoding.
#include "dataset/camera.h"
#include "dataset/mesh.h"
#include <string>
#include <vector>
#include <functional>
#include <cstdint>

namespace b2c {

// The `out/mesh` cameras format: width/height at the top, then per camera name, fx fy cx cy, position and the OpenGL
// camera-to-world rotation (body2colmap's cameras.json; `camera_from_json` in render/renderer.h does the conversion).
struct CamSet {
  int W = 0, H = 0;
  std::vector<Camera> cams;
  std::vector<std::string> names;   // the file names ("frame_00031_.png")
  size_t size() const { return cams.size(); }
};
CamSet load_cams(const std::string& path);
std::string file_stem(const std::string& name);   // "a/b.png" -> "b"

// 8-bit image as stb decodes it, forced to `channels` (1 = grey, 4 = RGBA).
struct Image8 { int W = 0, H = 0, C = 0; std::vector<uint8_t> px; bool ok() const { return W > 0; } };
Image8 load_image8(const std::string& path, int channels);       // fails when the file is missing
bool try_load_image8(const std::string& path, int channels, Image8& out);
// 16-bit greyscale PNG (the probe's <name>.zfirst.png, millimetres). Returns false when missing.
bool try_load_png16(const std::string& path, int& W, int& H, std::vector<uint16_t>& out);
void write_png16(const std::string& path, int W, int H, const uint16_t* px);
void write_png8(const std::string& path, int W, int H, int C, const uint8_t* px);
// Raw little-endian float32 maps (`<stem>.depth.f32` and friends): no header, the size comes from the consumer.
void write_f32(const std::string& path, const float* data, size_t n);
std::vector<float> read_f32(const std::string& path, size_t n_expected);
bool file_exists(const std::string& path);
// The xyz of a point set: a vertex-only ply (the face cap's Gaussians, through read_ply) or any mesh's vertices.
std::vector<float> read_points(const std::string& path);

// Runs fn(i) for i in [0, n) on a pool of threads (PNG decoding, adjacency building).
void parallel_for(size_t n, const std::function<void(size_t)>& fn, unsigned max_threads = 0);

// A small clap-like option parser for the subcommands: `--name VALUE`, `--flag`, repeatable multi-value options
// (`--views CAMS DIR`), and `--help` printing the table.
struct ArgParser {
  struct Opt { std::string name, value_name, help; int nvals; bool repeat; std::function<void(const std::vector<std::string>&)> set; int seen = 0; };
  std::string usage;
  std::vector<Opt> opts;
  explicit ArgParser(std::string usage_line) : usage(std::move(usage_line)) {}
  ArgParser& s(const char* name, const char* value, const char* help, std::string& out);
  ArgParser& f(const char* name, const char* value, const char* help, float& out);
  ArgParser& i(const char* name, const char* value, const char* help, int& out);
  ArgParser& b(const char* name, const char* help, bool& out);
  ArgParser& f3(const char* name, const char* value, const char* help, float* out);   // "x,y,z"
  // Repeatable: every occurrence appends its nvals values as one entry.
  ArgParser& multi(const char* name, const char* value, int nvals, const char* help, std::vector<std::vector<std::string>>& out);
  // Parses argv[start..]; positional arguments are not allowed. Returns false when --help was given (help printed).
  bool parse(int argc, char** argv, int start = 2);
  std::string help() const;
  bool given(const char* name) const;
};

}  // namespace b2c
