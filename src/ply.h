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

  int K() const { return sh_coeffs_for_degree(sh_degree); }
  void resize(size_t count, int degree);
};

extern const char* const EVIDENCE_FIELDS[7];

SplatCloud read_ply(const std::string& path);
void write_ply(const std::string& path, const SplatCloud& c, const std::vector<std::string>& comments);

}  // namespace b2c
