// lod-merge (lod/merge.h): the two-splat moment match and edge cost against splat-transform 3.10.1's own TypeScript
// (moment-match.ts mergeGroup, decimate-uniform/edge-cost-cpu.ts computeEdgeCostView, run under node on these
// inputs), and the merge rules: the count is hit exactly, no group crosses a class, seg_conf is the group minimum.
#include "lod/merge.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <set>

using namespace b2c;

namespace {
// TypeScript colour layout (dc0..2, then f_rest channel-major) -> SplatCloud lane (k * 3 + ch), SH degree 1.
int ts_lane(int t) { return t < 3 ? t : ((t - 3) % 3 + 1) * 3 + (t - 3) / 3; }
bool near(double a, double b, double tol, const char* what, int& fails) {
  if (std::abs(a - b) <= tol * std::max(1.0, std::abs(b))) return true;
  printf("  lod: %s = %.9g, splat-transform %.9g\n", what, a, b); fails++; return false;
}
}  // namespace

int test_lod() {
  int fails = 0;
  {
    SplatCloud c; c.resize(2, 1);
    const float pos[6] = {0.1f, -0.2f, 0.3f, 0.13f, -0.18f, 0.31f};
    const float geo[16] = {0.9f, 0.1f, -0.3f, 0.2f, std::log(0.02f), std::log(0.01f), std::log(0.005f), 0.7f,
                           0.5f, -0.4f, 0.6f, 0.1f, std::log(0.015f), std::log(0.03f), std::log(0.004f), -0.3f};
    for (int i = 0; i < 2; i++) {
      for (int a = 0; a < 3; a++) { c.pos[i * 3 + a] = pos[i * 3 + a]; c.log_scale[i * 3 + a] = geo[i * 8 + 4 + a]; }
      for (int a = 0; a < 4; a++) c.quat[i * 4 + a] = geo[i * 8 + a];
      c.opacity[i] = geo[i * 8 + 7];
      for (int t = 0; t < 12; t++) c.sh[i * 12 + ts_lane(t)] = (float)(std::sin((i * 12 + t) * 1.7) * (t < 3 ? 1.2 : 0.3));
    }
    SplatCloud out; out.resize(1, 1);
    const int m[2] = {0, 1};
    lod_merge_group(c, m, 2, out, 0);
    const double ref_pos[3] = {0.11669026446695971, -0.18887316099566986, 0.3055634287681377};
    const double ref_geo[8] = {0.9643517349617505, 0.0952692850922628, -0.24684059343819795, 0.004383608713020534, -3.592084631990348, -3.9727916950779387, -4.602326911709787, 0.20780934048589084};
    const double ref_col[12] = {0.6674724052575604, 0.45542483705979614, -0.7848303308848897, -0.06329567449033949, 0.21251818185922244, 0.008532074934987255, -0.21471681088907868, 0.04679808285589199, 0.20265745485477274, -0.09902068747876056, -0.1771409201046505, 0.14466793583238652};
    for (int a = 0; a < 3; a++) near(out.pos[a], ref_pos[a], 1e-6, "merged mean", fails);
    for (int a = 0; a < 4; a++) near(out.quat[a], ref_geo[a], 1e-5, "merged rotation", fails);
    for (int a = 0; a < 3; a++) near(out.log_scale[a], ref_geo[4 + a], 1e-5, "merged log scale", fails);
    near(out.opacity[0], ref_geo[7], 1e-5, "merged opacity logit", fails);
    for (int t = 0; t < 12; t++) near(out.sh[ts_lane(t)], ref_col[t], 1e-6, "merged SH", fails);
    near(lod_edge_cost(c, 0, 1), 5.008853576727031, 1e-5, "edge cost", fails);
  }
  {   // seg_conf is the group minimum, the class the members share
    SplatCloud c; c.has_labels = true; c.resize(3, 0);
    for (int i = 0; i < 3; i++) { c.pos[i * 3] = 0.01f * i; c.log_scale[i * 3] = c.log_scale[i * 3 + 1] = c.log_scale[i * 3 + 2] = std::log(0.01f); c.labels[i * 2] = 7; }
    c.labels[1] = 0.9f; c.labels[3] = 0.3f; c.labels[5] = 0.7f;
    SplatCloud out; out.has_labels = true; out.resize(1, 0);
    const int m[3] = {0, 1, 2};
    lod_merge_group(c, m, 3, out, 0);
    if (out.labels[0] != 7.f || out.labels[1] != 0.3f) { printf("  lod: merged label %g conf %g, expected 7 and 0.3\n", out.labels[0], out.labels[1]); fails++; }
  }
  {   // two classes interleaved in space, each a constant colour: a merge across them would mix the colours
    std::mt19937 rng(5); std::uniform_real_distribution<float> u(0.f, 1.f);
    const int n = 6000;
    SplatCloud c; c.has_labels = true; c.resize(n, 1);
    for (int i = 0; i < n; i++) {
      for (int a = 0; a < 3; a++) { c.pos[i * 3 + a] = u(rng); c.log_scale[i * 3 + a] = std::log(0.005f + 0.01f * u(rng)); }
      for (int a = 0; a < 4; a++) c.quat[i * 4 + a] = u(rng) - 0.5f;
      c.opacity[i] = 4.f * u(rng) - 2.f;
      const bool A = u(rng) < 0.6f;
      c.labels[i * 2] = A ? 2.f : 5.f; c.labels[i * 2 + 1] = 0.2f + 0.8f * u(rng);
      for (int k = 0; k < 12; k++) c.sh[i * 12 + k] = A ? 1.f : -1.f;
    }
    for (size_t count : {3001, 1500, 777, 40}) {
      MergeParams p; p.count = count;
      SplatCloud o = lod_merge(c, p);
      if (o.n != count) { printf("  lod: merged to %zu splats, asked for %zu\n", o.n, count); fails++; }
      int crossed = 0;
      for (size_t i = 0; i < o.n; i++) {
        const float want = o.labels[i * 2] == 2.f ? 1.f : -1.f;
        if ((o.labels[i * 2] != 2.f && o.labels[i * 2] != 5.f) || std::abs(o.sh[i * 12] - want) > 1e-5f) crossed++;
      }
      if (crossed) { printf("  lod: %d of %zu merged splats mix the two classes (count %zu)\n", crossed, o.n, count); fails++; }
      if (o.has_evidence || o.has_fill || o.has_gate || o.has_open) { printf("  lod: per-splat state survived the merge\n"); fails++; }
    }
  }
  printf("lod: %s\n", fails ? "FAILED" : "ok (merge and edge cost match splat-transform; exact count; no cross-class merge; min seg_conf)");
  return fails;
}
