#pragma once
#include <string>
#include <vector>
#include <cstddef>

namespace b2c {

constexpr float SH_C0 = 0.28209479177387814f;
inline int sh_coeffs_for_degree(int d) { return (d + 1) * (d + 1); }

// Host-side splat cloud in brush's parameterization.
struct SplatCloud {
  size_t n = 0;
  int sh_degree = 0;
  std::vector<float> pos;        // [n][3]
  std::vector<float> log_scale;  // [n][3]
  std::vector<float> quat;       // [n][4] (w, x, y, z)
  std::vector<float> opacity;    // [n] raw (logit)
  std::vector<float> sh;         // [n][K][3] coefficient-major
  bool has_scales = true;        // false when scales must be initialised (kNN)
  bool has_evidence = false;
  std::vector<float> evidence;   // [n][7]: w_in, w_all, err, views (effective count, see evidence.cu), dir xyz
  bool has_labels = false;
  std::vector<float> labels;     // [n][2]: seg_label (class id, integer-valued), seg_conf (winner's share, 0..1)
  bool has_fill = false;
  std::vector<float> fill;       // [n]: cage_fill; > 0 fades IN with cage stretch, < 0 fades out on its gate face (gpu/cage.h)
  bool has_gate = false;
  std::vector<float> gate;       // [n][2]: cage_gate_a / _b, a cage vertex pair whose distance ratio (posed / canonical) gates the splat's fade (-1 = its own face's stretch)
  bool has_gate_s = false;
  std::vector<float> gate_s;     // [n]: cage_gate_s, the pair's reference separation (m): the gate reads (posed - canonical distance) / gate_s instead of the ratio (0 = ratio)
  bool has_open = false;
  std::vector<float> open;       // [n][4]: open_r, open_g, open_b (colour 0..1), open_dopacity: the splat's second, "open crease" state (gpu/cage_open.h)

  int K() const { return sh_coeffs_for_degree(sh_degree); }
  void resize(size_t count, int degree);
};

extern const char* const EVIDENCE_FIELDS[7];
// Appended after the ev_* block; plain float properties so a viewer that reads brush's layout skips them.
extern const char* const LABEL_FIELDS[2];

// Header-only probe: true when the file is a ply read_ply can load, i.e. one whose only
// non-empty element is 'vertex'. A triangle mesh (the hollow proxy's mesh.ply) is not.
bool ply_is_vertex_only(const std::string& path);

SplatCloud read_ply(const std::string& path);
// The "comment ..." lines of a ply header (without the "comment " prefix).
std::vector<std::string> read_ply_comments(const std::string& path);
void write_ply(const std::string& path, const SplatCloud& c, const std::vector<std::string>& comments);

}  // namespace b2c
