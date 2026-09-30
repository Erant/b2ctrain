// Cage appearance MLP (gpu/cage_app.h): the per-vertex backward against central finite differences of
// L = sum_v sum_j c_vj R_vj, and R's blend / opacity / scale channels vanishing at the canonical pose.
#include "gpu/cage_app.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace b2c;

int test_cage_app() {
  int fails = 0;
  // A 5 x 5 vertex sheet, posed by a non-uniform stretch and a bend.
  const int G = 5, nv = G * G;
  std::vector<float3> v0, v1; std::vector<int3> fc;
  for (int y = 0; y < G; y++) for (int x = 0; x < G; x++) {
    v0.push_back(make_float3(0.01f * x, 0.01f * y, 0.f));
    float sx = x > 2 ? 0.01f * 2 + 0.01f * 1.9f * (x - 2) : 0.01f * x;
    v1.push_back(make_float3(sx, 0.01f * y * (1.f + 0.1f * x), 0.004f * x * x));
  }
  for (int y = 0; y + 1 < G; y++) for (int x = 0; x + 1 < G; x++) {
    int a = y * G + x, b = a + 1, c = a + G, d = c + 1;
    fc.push_back(make_int3(a, b, d)); fc.push_back(make_int3(a, d, c));
  }
  const int nf = (int)fc.size();
  std::string path = "/tmp/b2c_cage_app_test.b2ccage";
  {
    FILE* f = fopen(path.c_str(), "wb");
    int32_t hdr[4] = {1, nv, nf, 2}; int32_t lay[4] = {0, nv, 0, nf}; uint32_t bits = 0xFFFFFFFFu;
    fwrite("B2CCAGE1", 1, 8, f); fwrite(hdr, 4, 4, f); fwrite(lay, 4, 4, f); fwrite(&bits, 4, 1, f);
    fwrite(v0.data(), sizeof(float3), nv, f); fwrite(fc.data(), sizeof(int3), nf, f);
    char nm[128] = {0}; strcpy(nm, "rest"); strcpy(nm + 64, "posed"); fwrite(nm, 1, 128, f);
    fwrite(v0.data(), sizeof(float3), nv, f); fwrite(v1.data(), sizeof(float3), nv, f);
    fclose(f);
  }
  CageRig cage; cage.load(path);
  CageApp app; app.init(cage, 7, 0);
  // Larger output weights than the init's, so every channel carries signal.
  {
    std::vector<float> P = app.P.download(APP_NP);
    std::mt19937 rng(3); std::normal_distribution<float> nd(0.f, 0.3f);
    for (int k = APP_NP - APP_OUT * (APP_HID + 1); k < APP_NP; k++) P[k] += nd(rng);
    app.P.upload(P);
  }
  auto set_frame = [&](const std::vector<float3>& v) { cage.verts_view.upload(v); };
  // Rest: R0, R4, R5 vanish.
  set_frame(v0); app.vert_forward(cage, 0);
  {
    auto R = app.vout.download((size_t)nv * APP_OUT); double mx = 0;
    for (int k = 0; k < nv; k++) for (int j : {0, 4, 5, 6, 7, 8}) mx = std::max(mx, (double)std::abs(R[(size_t)k * APP_OUT + j]));
    if (mx > 1e-6) { fails++; printf("cage_app: rest pose output %.3g, expected 0\n", mx); }
  }
  set_frame(v1);
  std::mt19937 rng(11); std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> c((size_t)nv * APP_OUT); for (auto& x : c) x = nd(rng);
  auto loss = [&]() {
    app.vert_forward(cage, 0);
    auto R = app.vout.download((size_t)nv * APP_OUT); double L = 0;
    for (size_t k = 0; k < R.size(); k++) L += (double)c[k] * R[k];
    return L;
  };
  app.vert_forward(cage, 0);
  {
    auto feat = app.vfeat.download((size_t)nv * APP_FEAT); double mx = 0;
    for (float f : feat) mx = std::max(mx, (double)std::abs(f));
    if (mx < 0.05) { fails++; printf("cage_app: posed features too small (%.3g)\n", mx); }
  }
  app.vert_forward(cage, 0);
  app.gP.zero(0); app.gZ.zero(0);
  app.g_vout.upload(c);
  app.vert_backward(0);
  std::vector<float> gP = app.gP.download(APP_NP), gZ = app.gZ.download((size_t)nv * APP_LAT);
  std::vector<float> P = app.P.download(APP_NP), Z = app.Z.download((size_t)nv * APP_LAT);
  double max_rel = 0; int checked = 0;
  auto check = [&](std::vector<float>& buf, DevBuf<float>& dev, const std::vector<float>& an, int k, const char* what) {
    const float eps = 1e-2f, orig = buf[k];
    buf[k] = orig + eps; dev.upload(buf); double lp = loss();
    buf[k] = orig - eps; dev.upload(buf); double lm = loss();
    buf[k] = orig; dev.upload(buf);
    double fd = (lp - lm) / (2.0 * eps), a = an[k];
    double rel = std::abs(fd - a) / std::max(std::max(std::abs(fd), std::abs(a)), 1e-4);
    max_rel = std::max(max_rel, rel); checked++;
    if (rel > 0.02 && std::abs(fd - a) > 1e-4) { fails++; printf("  cage_app MISMATCH %s[%d]: analytic %.6g fd %.6g\n", what, k, a, fd); }
  };
  for (int k = 0; k < APP_NP; k += 7) check(P, app.P, gP, k, "P");
  for (int k = 0; k < nv * APP_LAT; k += 5) check(Z, app.Z, gZ, k, "Z");
  printf("cage_app: checked %d parameters, max rel err %.4f %s\n", checked, max_rel, fails ? "FAILED" : "ok");
  return fails;
}
