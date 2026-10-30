// lod-merge: see merge.h. The cost, selection and merge functions follow splat-transform 3.10.1 line by line
// (PlayCanvas Ltd., MIT License); the comments name the TypeScript each one ports.
#include "lod/merge.h"
#include "util/log.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <thread>

namespace b2c {
namespace {

constexpr double EPS_COV = 1e-8;
constexpr double LOG2PI = 1.8378770664093453;   // log(2 pi)
constexpr double PI = 3.14159265358979323846;
constexpr int KNN_K = 16;                       // decimate-uniform/decimate-source.ts
constexpr int CAND_K = 4;                       // chooseK: 4 whenever the resident estimate fits (always, at our sizes)
constexpr int SELECT_BUCKETS = 1024;            // decimate-uniform/select.ts
constexpr double MIN_ITERATION_PROGRESS = 0.05;
constexpr int RING_MAX = 32;                    // as gpu/cage.cu's binding

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double logit(double p) { p = std::max(1e-7, std::min(1.0 - 1e-7, p)); return std::log(p / (1.0 - p)); }
double log_add_exp(double a, double b) {
  if (a == -INFINITY) return b;
  if (b == -INFINITY) return a;
  double m = std::max(a, b);
  return m + std::log(std::exp(a - m) + std::exp(b - m));
}

// moment-match.ts: makeGaussianSamples(1, 0), mulberry32 seeded 0.
void mc_sample(double z[3]) {
  uint32_t seed = 0;
  auto rand = [&]() {
    uint32_t t = (seed += 0x6d2b79f5u);
    t = (t ^ (t >> 15)) * (t | 1u);
    t ^= t + (t ^ (t >> 7)) * (t | 61u);
    return (double)(t ^ (t >> 14)) / 4294967296.0;
  };
  double u1 = std::max(rand(), 1e-12), u2 = rand(), u3 = std::max(rand(), 1e-12), u4 = rand();
  double r1 = std::sqrt(-2 * std::log(u1)), t1 = 2 * PI * u2, r2 = std::sqrt(-2 * std::log(u3)), t2 = 2 * PI * u4;
  z[0] = r1 * std::cos(t1); z[1] = r1 * std::sin(t1); z[2] = r2 * std::cos(t2);
}

void quat_to_rotmat(double qw, double qx, double qy, double qz, double* R) {
  double xx = qx * qx, yy = qy * qy, zz = qz * qz, wx = qw * qx, wy = qw * qy, wz = qw * qz, xy = qx * qy, xz = qx * qz, yz = qy * qz;
  R[0] = 1 - 2 * (yy + zz); R[1] = 2 * (xy - wz); R[2] = 2 * (xz + wy);
  R[3] = 2 * (xy + wz); R[4] = 1 - 2 * (xx + zz); R[5] = 2 * (yz - wx);
  R[6] = 2 * (xz - wy); R[7] = 2 * (yz + wx); R[8] = 1 - 2 * (xx + yy);
}
void sigma_from_rot_var(const double* R, double vx, double vy, double vz, double* S) {
  S[0] = R[0] * R[0] * vx + R[1] * R[1] * vy + R[2] * R[2] * vz;
  S[1] = R[0] * R[3] * vx + R[1] * R[4] * vy + R[2] * R[5] * vz;
  S[2] = R[0] * R[6] * vx + R[1] * R[7] * vy + R[2] * R[8] * vz;
  S[3] = S[1];
  S[4] = R[3] * R[3] * vx + R[4] * R[4] * vy + R[5] * R[5] * vz;
  S[5] = R[3] * R[6] * vx + R[4] * R[7] * vy + R[5] * R[8] * vz;
  S[6] = S[2]; S[7] = S[5];
  S[8] = R[6] * R[6] * vx + R[7] * R[7] * vy + R[8] * R[8] * vz;
}
double det3(const double* A) {
  return A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
}
double gauss_logpdf_diagrot(double x, double y, double z, double mx, double my, double mz, const double* R,
                            double ix, double iy, double iz, double logdet) {
  double dx = x - mx, dy = y - my, dz = z - mz;
  double y0 = dx * R[0] + dy * R[3] + dz * R[6], y1 = dx * R[1] + dy * R[4] + dz * R[7], y2 = dx * R[2] + dy * R[5] + dz * R[8];
  return -0.5 * (3 * LOG2PI + logdet + y0 * y0 * ix + y1 * y1 * iy + y2 * y2 * iz);
}
// Jacobi: eigenvalues on A's diagonal, eigenvectors in V's columns (moment-match.ts eigenSymmetric3x3).
void eigen_sym3(const double* Ain, double* A, double* V) {
  std::copy(Ain, Ain + 9, A);
  for (int k = 0; k < 9; k++) V[k] = (k % 4 == 0) ? 1 : 0;
  for (int iter = 0; iter < 24; iter++) {
    int p = 0, q = 1; double mx = std::abs(A[1]);
    if (std::abs(A[2]) > mx) { p = 0; q = 2; mx = std::abs(A[2]); }
    if (std::abs(A[5]) > mx) { p = 1; q = 2; mx = std::abs(A[5]); }
    if (mx < 1e-12) break;
    int pp = 3 * p + p, qq = 3 * q + q, pq = 3 * p + q;
    double app = A[pp], aqq = A[qq], apq = A[pq];
    double tau = (aqq - app) / (2 * apq);
    double sg = tau > 0 ? 1.0 : (tau < 0 ? -1.0 : 0.0);   // Math.sign
    double t = sg / (std::abs(tau) + std::sqrt(1 + tau * tau));
    double c = 1 / std::sqrt(1 + t * t), s = t * c;
    for (int k = 0; k < 3; k++) {
      if (k == p || k == q) continue;
      int kp = 3 * k + p, kq = 3 * k + q, pk = 3 * p + k, qk = 3 * q + k;
      double akp = A[kp], akq = A[kq];
      A[kp] = c * akp - s * akq; A[pk] = A[kp];
      A[kq] = s * akp + c * akq; A[qk] = A[kq];
    }
    A[pp] = c * c * app - 2 * s * c * apq + s * s * aqq;
    A[qq] = s * s * app + 2 * s * c * apq + c * c * aqq;
    A[pq] = 0; A[3 * q + p] = 0;
    for (int k = 0; k < 3; k++) {
      int kp = 3 * k + p, kq = 3 * k + q;
      double vkp = V[kp], vkq = V[kq];
      V[kp] = c * vkp - s * vkq; V[kq] = s * vkp + c * vkq;
    }
  }
}
void rotmat_to_quat(const double* R, double* q) {
  double tr = R[0] + R[4] + R[8], qw, qx, qy, qz;
  if (tr > 0) { double S = std::sqrt(tr + 1) * 2; qw = 0.25 * S; qx = (R[7] - R[5]) / S; qy = (R[2] - R[6]) / S; qz = (R[3] - R[1]) / S; }
  else if (R[0] > R[4] && R[0] > R[8]) { double S = std::sqrt(1 + R[0] - R[4] - R[8]) * 2; qw = (R[7] - R[5]) / S; qx = 0.25 * S; qy = (R[1] + R[3]) / S; qz = (R[2] + R[6]) / S; }
  else if (R[4] > R[8]) { double S = std::sqrt(1 + R[4] - R[0] - R[8]) * 2; qw = (R[2] - R[6]) / S; qx = (R[1] + R[3]) / S; qy = 0.25 * S; qz = (R[5] + R[7]) / S; }
  else { double S = std::sqrt(1 + R[8] - R[0] - R[4]) * 2; qw = (R[3] - R[1]) / S; qx = (R[2] + R[6]) / S; qy = (R[5] + R[7]) / S; qz = 0.25 * S; }
  double inv = 1 / std::max(std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz), 1e-12);
  q[0] = qw * inv; q[1] = qx * inv; q[2] = qy * inv; q[3] = qz * inv;
}
// Knud Thomsen's ellipsoid surface area, p = 1.6075.
double ellipsoid_area(double sx, double sy, double sz) {
  const double P = 1.6075;
  double a = std::pow(sx * sy, P), b = std::pow(sx * sz, P), c = std::pow(sy * sz, P);
  return 4 * PI * std::pow((a + b + c) / 3, 1 / P);
}
// Largest eigenvalue first (Smith's closed form, decimate/edge-cost-cpu.ts).
void eig_smith(double sxx, double sxy, double sxz, double syy, double syz, double szz, double& e0, double& e1, double& e2) {
  double q = (sxx + syy + szz) / 3, p1 = sxy * sxy + sxz * sxz + syz * syz;
  if (p1 <= 1e-30) { e0 = sxx; e1 = syy; e2 = szz; if (e0 < e1) std::swap(e0, e1); if (e0 < e2) std::swap(e0, e2); return; }
  double p2 = (sxx - q) * (sxx - q) + (syy - q) * (syy - q) + (szz - q) * (szz - q) + 2 * p1;
  double p = std::sqrt(p2 / 6), ip = 1 / p;
  double b00 = (sxx - q) * ip, b11 = (syy - q) * ip, b22 = (szz - q) * ip, b01 = sxy * ip, b02 = sxz * ip, b12 = syz * ip;
  double r = (b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02)) / 2;
  r = std::max(-1.0, std::min(1.0, r));
  double phi = std::acos(r) / 3;
  e0 = q + 2 * p * std::cos(phi); e2 = q + 2 * p * std::cos(phi + 2 * PI / 3); e1 = 3 * q - e0 - e2;
}

// The per-splat quantities of one generation.
struct View {
  const SplatCloud* c;
  int K3;   // SH floats per splat
  double mass(size_t i) const {   // splatMass: area * alpha (+1e-30, the merge path's epsilon)
    double sx = std::max(std::exp((double)c->log_scale[i * 3]), 1e-12), sy = std::max(std::exp((double)c->log_scale[i * 3 + 1]), 1e-12), sz = std::max(std::exp((double)c->log_scale[i * 3 + 2]), 1e-12);
    return sigmoid(c->opacity[i]) * ellipsoid_area(sx, sy, sz) + 1e-30;
  }
  // Σ_i = R diag(s^2) R^T from the normalised quaternion.
  void sigma(size_t i, double* S, double* R = nullptr, double add = 0) const {
    double qw = c->quat[i * 4], qx = c->quat[i * 4 + 1], qy = c->quat[i * 4 + 2], qz = c->quat[i * 4 + 3];
    double qn = 1 / std::max(std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz), 1e-12);
    double Rl[9]; double* Rr = R ? R : Rl;
    quat_to_rotmat(qw * qn, qx * qn, qy * qn, qz * qn, Rr);
    double sx = std::max(std::exp((double)c->log_scale[i * 3]), 1e-12), sy = std::max(std::exp((double)c->log_scale[i * 3 + 1]), 1e-12), sz = std::max(std::exp((double)c->log_scale[i * 3 + 2]), 1e-12);
    sigma_from_rot_var(Rr, sx * sx + add, sy * sy + add, sz * sz + add, S);
  }
};

// Weights (normalised), total mass, mean and covariance of a group: the first half of mergeGroup.
void group_moments(const View& v, const int* m, int n, double* w, double& W, double mu[3], double S[9]) {
  W = 0;
  for (int k = 0; k < n; k++) { w[k] = v.mass(m[k]); W += w[k]; }
  for (int k = 0; k < n; k++) w[k] /= W;
  mu[0] = mu[1] = mu[2] = 0;
  for (int k = 0; k < n; k++) for (int a = 0; a < 3; a++) mu[a] += w[k] * v.c->pos[(size_t)m[k] * 3 + a];
  std::fill(S, S + 9, 0.0);
  for (int k = 0; k < n; k++) {
    double Si[9]; v.sigma(m[k], Si);
    const float* p = &v.c->pos[(size_t)m[k] * 3];
    double dx = p[0] - mu[0], dy = p[1] - mu[1], dz = p[2] - mu[2];
    S[0] += w[k] * (dx * dx + Si[0]); S[1] += w[k] * (dx * dy + Si[1]); S[2] += w[k] * (dx * dz + Si[2]);
    S[4] += w[k] * (dy * dy + Si[4]); S[5] += w[k] * (dy * dz + Si[5]); S[8] += w[k] * (dz * dz + Si[8]);
  }
  S[3] = S[1]; S[6] = S[2]; S[7] = S[5];
  S[0] += EPS_COV; S[4] += EPS_COV; S[8] += EPS_COV;
}

// ---- edge costs ----

// decimate-uniform/edge-cost-cpu.ts: buildCostCache + computeEdgeCostView (one MC sample, seed 0).
struct KlCache {
  std::vector<double> R, var, inv, logdet, sigma, mass;
  double z[3];
  void build(const View& v) {
    size_t n = v.c->n;
    R.resize(n * 9); var.resize(n * 3); inv.resize(n * 3); logdet.resize(n); sigma.resize(n * 9); mass.resize(n);
    mc_sample(z);
    for (size_t i = 0; i < n; i++) {
      double sx = std::max(std::exp((double)v.c->log_scale[i * 3]), 1e-12), sy = std::max(std::exp((double)v.c->log_scale[i * 3 + 1]), 1e-12), sz = std::max(std::exp((double)v.c->log_scale[i * 3 + 2]), 1e-12);
      double vx = sx * sx + EPS_COV, vy = sy * sy + EPS_COV, vz = sz * sz + EPS_COV;
      var[i * 3] = vx; var[i * 3 + 1] = vy; var[i * 3 + 2] = vz;
      inv[i * 3] = 1 / std::max(vx, 1e-30); inv[i * 3 + 1] = 1 / std::max(vy, 1e-30); inv[i * 3 + 2] = 1 / std::max(vz, 1e-30);
      logdet[i] = std::log(std::max(vx, 1e-30)) + std::log(std::max(vy, 1e-30)) + std::log(std::max(vz, 1e-30));
      v.sigma(i, &sigma[i * 9], &R[i * 9], EPS_COV);
      mass[i] = sigmoid(v.c->opacity[i]) * ellipsoid_area(sx, sy, sz) + 1e-12;
    }
  }
  double cost(const View& v, size_t i, size_t j) const {
    const float* P = v.c->pos.data();
    double mux = P[i * 3], muy = P[i * 3 + 1], muz = P[i * 3 + 2], mvx = P[j * 3], mvy = P[j * 3 + 1], mvz = P[j * 3 + 2];
    double wi = mass[i], wj = mass[j], W = wi + wj, Ws = W > 0 ? W : 1;
    double pi = std::max(1e-12, std::min(1 - 1e-12, wi / Ws)), pj = 1 - pi;
    double lpi = std::log(pi), lpj = std::log(pj);
    double mmx = pi * mux + pj * mvx, mmy = pi * muy + pj * mvy, mmz = pi * muz + pj * mvz;
    double di[3] = {mux - mmx, muy - mmy, muz - mmz}, dj[3] = {mvx - mmx, mvy - mmy, mvz - mmz};
    double sm[9];
    for (int a = 0; a < 9; a++) sm[a] = pi * sigma[i * 9 + a] + pj * sigma[j * 9 + a];
    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) sm[r * 3 + c] += pi * di[r] * di[c] + pj * dj[r] * dj[c];
    sm[1] = sm[3] = 0.5 * (sm[1] + sm[3]); sm[2] = sm[6] = 0.5 * (sm[2] + sm[6]); sm[5] = sm[7] = 0.5 * (sm[5] + sm[7]);
    sm[0] += EPS_COV; sm[4] += EPS_COV; sm[8] += EPS_COV;
    double logdetm = std::log(std::max(det3(sm), 1e-30));
    double EpNegLogQ = 0.5 * (3 * LOG2PI + logdetm + 3);
    const double* Ri = &R[i * 9]; const double* Rj = &R[j * 9];
    double si[3] = {std::sqrt(var[i * 3]), std::sqrt(var[i * 3 + 1]), std::sqrt(var[i * 3 + 2])};
    double sj[3] = {std::sqrt(var[j * 3]), std::sqrt(var[j * 3 + 1]), std::sqrt(var[j * 3 + 2])};
    double xi[3], xj[3];
    for (int r = 0; r < 3; r++) {
      xi[r] = (r == 0 ? mux : r == 1 ? muy : muz) + z[0] * si[0] * Ri[r * 3] + z[1] * si[1] * Ri[r * 3 + 1] + z[2] * si[2] * Ri[r * 3 + 2];
      xj[r] = (r == 0 ? mvx : r == 1 ? mvy : mvz) + z[0] * sj[0] * Rj[r * 3] + z[1] * sj[1] * Rj[r * 3 + 1] + z[2] * sj[2] * Rj[r * 3 + 2];
    }
    auto lp = [&](const double* x, size_t k, double mx, double my, double mz) { return gauss_logpdf_diagrot(x[0], x[1], x[2], mx, my, mz, &R[k * 9], inv[k * 3], inv[k * 3 + 1], inv[k * 3 + 2], logdet[k]); };
    double Ei = log_add_exp(lpi + lp(xi, i, mux, muy, muz), lpj + lp(xi, j, mvx, mvy, mvz));
    double Ej = log_add_exp(lpi + lp(xj, i, mux, muy, muz), lpj + lp(xj, j, mvx, mvy, mvz));
    double geo = pi * Ei + pj * Ej + EpNegLogQ;
    double csh = 0;
    const float* a = &v.c->sh[i * v.K3]; const float* b = &v.c->sh[j * v.K3];
    for (int k = 0; k < v.K3; k++) { double d = (double)a[k] - b[k]; csh += d * d; }
    return geo + csh;
  }
};

// ---- exact k nearest neighbours (a static kd-tree over a subset of the points) ----
struct KdTree {
  const float* p = nullptr;
  std::vector<int> idx;
  struct Node { int lo, hi, left, right, axis; float split; };
  std::vector<Node> nodes;
  void build(const float* pts, std::vector<int> ids) {
    p = pts; idx = std::move(ids); nodes.clear();
    if (!idx.empty()) { nodes.reserve(2 * idx.size() / 8 + 2); rec(0, (int)idx.size()); }
  }
  int rec(int lo, int hi) {
    int id = (int)nodes.size(); nodes.push_back({lo, hi, -1, -1, -1, 0.f});
    if (hi - lo <= 8) return id;
    float mn[3] = {INFINITY, INFINITY, INFINITY}, mx[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (int k = lo; k < hi; k++) for (int a = 0; a < 3; a++) { float v = p[(size_t)idx[k] * 3 + a]; mn[a] = std::min(mn[a], v); mx[a] = std::max(mx[a], v); }
    int axis = 0; for (int a = 1; a < 3; a++) if (mx[a] - mn[a] > mx[axis] - mn[axis]) axis = a;
    int mid = (lo + hi) / 2;
    std::nth_element(idx.begin() + lo, idx.begin() + mid, idx.begin() + hi, [&](int a, int b) { return p[(size_t)a * 3 + axis] < p[(size_t)b * 3 + axis]; });
    float split = p[(size_t)idx[mid] * 3 + axis];
    int l = rec(lo, mid), r = rec(mid, hi);
    nodes[id].left = l; nodes[id].right = r; nodes[id].axis = axis; nodes[id].split = split;
    return id;
  }
  // The k nearest points to q other than `self`, ascending by distance (max-heap scratch in `h`).
  void knn(const float* q, int k, int self, std::vector<std::pair<float, int>>& h) const {
    h.clear();
    if (nodes.empty() || k <= 0) return;
    search(0, q, k, self, h);
    std::sort_heap(h.begin(), h.end());
  }
  void search(int ni, const float* q, int k, int self, std::vector<std::pair<float, int>>& h) const {
    const Node& n = nodes[ni];
    if (n.axis < 0) {
      for (int t = n.lo; t < n.hi; t++) {
        int j = idx[t]; if (j == self) continue;
        float dx = p[(size_t)j * 3] - q[0], dy = p[(size_t)j * 3 + 1] - q[1], dz = p[(size_t)j * 3 + 2] - q[2];
        float d = dx * dx + dy * dy + dz * dz;
        if ((int)h.size() < k) { h.emplace_back(d, j); std::push_heap(h.begin(), h.end()); }
        else if (d < h.front().first) { std::pop_heap(h.begin(), h.end()); h.back() = {d, j}; std::push_heap(h.begin(), h.end()); }
      }
      return;
    }
    float diff = q[n.axis] - n.split;
    int near = diff < 0 ? n.left : n.right, far = diff < 0 ? n.right : n.left;
    search(near, q, k, self, h);
    if ((int)h.size() < k || diff * diff <= h.front().first) search(far, q, k, self, h);
  }
};

// ---- the cage cap ----
double closest_on_tri_d2(const double* p, const float* a, const float* b, const float* c) {
  double ab[3], ac[3], ap[3], bp[3], cp[3];
  for (int k = 0; k < 3; k++) { ab[k] = b[k] - a[k]; ac[k] = c[k] - a[k]; ap[k] = p[k] - a[k]; bp[k] = p[k] - b[k]; cp[k] = p[k] - c[k]; }
  auto dot = [](const double* x, const double* y) { return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]; };
  double d1 = dot(ab, ap), d2 = dot(ac, ap), b1, b2;
  if (d1 <= 0 && d2 <= 0) { b1 = 0; b2 = 0; }
  else {
    double d3 = dot(ab, bp), d4 = dot(ac, bp), d5 = dot(ab, cp), d6 = dot(ac, cp);
    double vc = d1 * d4 - d3 * d2, vb = d5 * d2 - d1 * d6, va = d3 * d6 - d5 * d4;
    if (d3 >= 0 && d4 <= d3) { b1 = 1; b2 = 0; }
    else if (d6 >= 0 && d5 <= d6) { b1 = 0; b2 = 1; }
    else if (vc <= 0 && d1 >= 0 && d3 <= 0) { b1 = d1 / (d1 - d3); b2 = 0; }
    else if (vb <= 0 && d2 >= 0 && d6 <= 0) { b1 = 0; b2 = d2 / (d2 - d6); }
    else if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) { double t = (d4 - d3) / ((d4 - d3) + (d5 - d6)); b1 = 1 - t; b2 = t; }
    else { double den = 1 / (va + vb + vc); b1 = vb * den; b2 = vc * den; }
  }
  double s = 0;
  for (int k = 0; k < 3; k++) { double q = a[k] + ab[k] * b1 + ac[k] * b2 - p[k]; s += q * q; }
  return s;
}

// The face a splat at p with class `label` would bind to (gpu/cage.cu: nearest vertex of the layer that owns the
// class, else of any layer, then the nearest of the triangles around it), and its mean edge length.
struct CapIndex {
  const CageFaces* cf = nullptr;
  std::vector<KdTree> trees;          // per layer, plus one over every layer's vertices last
  std::vector<int> vf_off, vf_idx;
  std::vector<double> edge;
  void build(const CageFaces& c) {
    cf = &c;
    const int nv = (int)c.verts.size() / 3, nf = (int)c.faces.size() / 3, nl = (int)c.layer_classes.size();
    vf_off.assign(nv + 1, 0);
    for (int x : c.faces) vf_off[x + 1]++;
    for (int k = 0; k < nv; k++) vf_off[k + 1] += vf_off[k];
    vf_idx.resize(vf_off[nv]);
    std::vector<int> fill(vf_off.begin(), vf_off.end() - 1);
    for (int f = 0; f < nf; f++) for (int k = 0; k < 3; k++) vf_idx[fill[c.faces[f * 3 + k]]++] = f;
    edge.resize(nf);
    for (int f = 0; f < nf; f++) {
      const float* v[3] = {&c.verts[c.faces[f * 3] * 3], &c.verts[c.faces[f * 3 + 1] * 3], &c.verts[c.faces[f * 3 + 2] * 3]};
      double s = 0;
      for (int e = 0; e < 3; e++) { const float* a = v[e]; const float* b = v[(e + 1) % 3]; s += std::sqrt((double)(a[0] - b[0]) * (a[0] - b[0]) + (double)(a[1] - b[1]) * (a[1] - b[1]) + (double)(a[2] - b[2]) * (a[2] - b[2])); }
      edge[f] = s / 3;
    }
    trees.resize(nl + 1);
    std::vector<int> all;
    for (int l = 0; l < nl; l++) {
      std::vector<int> ids;
      for (int k = c.layer_v_off[l]; k < c.layer_v_off[l] + c.layer_v_count[l]; k++) if (vf_off[k + 1] > vf_off[k]) { ids.push_back(k); all.push_back(k); }
      trees[l].build(c.verts.data(), std::move(ids));
    }
    trees[nl].build(c.verts.data(), std::move(all));
  }
  double face_edge(const double p[3], int label) const {
    const int nl = (int)cf->layer_classes.size();
    int owner = nl;
    if (label >= 0 && label < 32) for (int l = 0; l < nl; l++) if (cf->layer_classes[l] & (1u << label)) { owner = l; break; }
    std::vector<std::pair<float, int>> h;
    float q[3] = {(float)p[0], (float)p[1], (float)p[2]};
    trees[owner].knn(q, 1, -1, h);
    if (h.empty()) return INFINITY;
    int v = h[0].second, best = -1; double bd = INFINITY;
    for (int k = vf_off[v]; k < std::min(vf_off[v + 1], vf_off[v] + RING_MAX); k++) {
      int f = vf_idx[k];
      double d2 = closest_on_tri_d2(p, &cf->verts[cf->faces[f * 3] * 3], &cf->verts[cf->faces[f * 3 + 1] * 3], &cf->verts[cf->faces[f * 3 + 2] * 3]);
      if (d2 < bd) { bd = d2; best = f; }
    }
    return best >= 0 ? edge[best] : INFINITY;
  }
};

void parallel_for(size_t n, const std::function<void(size_t, size_t)>& fn) {
  unsigned nt = std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
  if (n < 4096) nt = 1;
  std::vector<std::thread> th;
  size_t per = (n + nt - 1) / nt;
  for (unsigned t = 0; t < nt; t++) { size_t a = t * per, b = std::min(n, a + per); if (a < b) th.emplace_back(fn, a, b); }
  for (auto& t : th) t.join();
}

int label_of(const SplatCloud& c, size_t i) { return c.has_labels ? (int)std::lround(c.labels[i * 2]) : 0; }

// The splat and its labels only (merge.h: evidence and b2crig state are dropped).
SplatCloud core_copy(const SplatCloud& in) {
  SplatCloud c; c.has_labels = in.has_labels; c.has_scales = true;
  c.n = in.n; c.sh_degree = in.sh_degree;
  c.pos = in.pos; c.log_scale = in.log_scale; c.quat = in.quat; c.opacity = in.opacity; c.sh = in.sh;
  if (in.has_labels) c.labels = in.labels;
  return c;
}

}  // namespace

void lod_merge_group(const SplatCloud& c, const int* m, int n, SplatCloud& out, size_t o) {
  View v{&c, c.K() * 3};
  std::vector<double> w(n);
  double W, mu[3], S[9];
  group_moments(v, m, n, w.data(), W, mu, S);
  double A[9], V[9];
  eigen_sym3(S, A, V);
  double v0 = A[0], v1 = A[4], v2 = A[8];
  int o0, o1, o2;
  if (v0 >= v1) { if (v1 >= v2) { o0 = 0; o1 = 1; o2 = 2; } else if (v0 >= v2) { o0 = 0; o1 = 2; o2 = 1; } else { o0 = 2; o1 = 0; o2 = 1; } }
  else { if (v0 >= v2) { o0 = 1; o1 = 0; o2 = 2; } else if (v1 >= v2) { o0 = 1; o1 = 2; o2 = 0; } else { o0 = 2; o1 = 1; o2 = 0; } }
  double s0 = std::sqrt(std::max(A[3 * o0 + o0], 1e-18)), s1 = std::sqrt(std::max(A[3 * o1 + o1], 1e-18)), s2 = std::sqrt(std::max(A[3 * o2 + o2], 1e-18));
  double alpha = std::min(1.0, W / std::max(ellipsoid_area(s0, s1, s2), 1e-30));
  double Rm[9] = {V[o0], V[o1], V[o2], V[3 + o0], V[3 + o1], V[3 + o2], V[6 + o0], V[6 + o1], V[6 + o2]};
  if (det3(Rm) < 0) { Rm[2] *= -1; Rm[5] *= -1; Rm[8] *= -1; }
  double q[4]; rotmat_to_quat(Rm, q);
  for (int a = 0; a < 3; a++) out.pos[o * 3 + a] = (float)mu[a];
  for (int a = 0; a < 4; a++) out.quat[o * 4 + a] = (float)q[a];
  out.log_scale[o * 3] = (float)std::log(s0); out.log_scale[o * 3 + 1] = (float)std::log(s1); out.log_scale[o * 3 + 2] = (float)std::log(s2);
  out.opacity[o] = (float)logit(std::max(0.0, std::min(1.0, alpha)));
  const int K3 = v.K3;
  for (int k = 0; k < K3; k++) { double acc = 0; for (int t = 0; t < n; t++) acc += w[t] * c.sh[(size_t)m[t] * K3 + k]; out.sh[o * K3 + k] = (float)acc; }
  if (out.has_labels && c.has_labels) {
    float conf = c.labels[(size_t)m[0] * 2 + 1];
    for (int t = 1; t < n; t++) conf = std::min(conf, c.labels[(size_t)m[t] * 2 + 1]);
    out.labels[o * 2] = c.labels[(size_t)m[0] * 2]; out.labels[o * 2 + 1] = conf;
  }
}

double lod_edge_cost(const SplatCloud& c, size_t i, size_t j) {
  View v{&c, c.K() * 3};
  KlCache kl; kl.build(v);
  return kl.cost(v, i, j);
}

SplatCloud lod_merge(const SplatCloud& in, const MergeParams& p, MergeStats* stats) {
  MergeStats st;
  if (p.count < 1) fail("lod-merge: the target count must be at least 1");
  SplatCloud cur = core_copy(in);
  {
    std::map<int, size_t> cls; for (size_t i = 0; i < cur.n; i++) cls[label_of(cur, i)]++;
    st.classes = cls.size();
  }
  CapIndex cap; if (p.cage) cap.build(*p.cage);
  while (cur.n > p.count) {
    const size_t N = cur.n;
    st.generations++;
    double t0 = now_seconds();
    View v{&cur, cur.K() * 3};
    KlCache kl; kl.build(v);

    // Neighbours within the class: one kd-tree per seg_label.
    std::map<int, std::vector<int>> by_class;
    for (size_t i = 0; i < N; i++) by_class[label_of(cur, i)].push_back((int)i);
    std::map<int, KdTree> trees;
    for (auto& [c, ids] : by_class) trees[c].build(cur.pos.data(), ids);
    // The cage cap: a group may not span more than K x the mean edge of the cage face nearest its mean.
    auto cap_ok = [&](const int* m, int n) {
      double w[8], W, mu[3], S[9];
      group_moments(v, m, n, w, W, mu, S);
      double e0, e1, e2; eig_smith(S[0], S[1], S[2], S[4], S[5], S[8], e0, e1, e2);
      return 3 * std::sqrt(std::max(e0, 0.0)) <= p.max_extent_faces * cap.face_edge(mu, label_of(cur, m[0]));
    };
    std::vector<int> cand_idx(N * CAND_K, -1);
    std::vector<float> cand_cost(N * CAND_K, INFINITY);
    std::vector<uint8_t> capped(N, 0);              // a merge of this splat was ruled out by the cap
    std::vector<size_t> blocked_pairs(N, 0);
    parallel_for(N, [&](size_t a, size_t b) {
      std::vector<std::pair<float, int>> h;
      for (size_t i = a; i < b; i++) {
        const KdTree& t = trees.at(label_of(cur, i));
        const int k = std::min<int>(KNN_K, (int)t.idx.size() - 1);
        t.knn(&cur.pos[i * 3], k, (int)i, h);
        // decimate-uniform/priority.ts: reduce to the best CAND_K, ascending (ties keep the nearer neighbour first).
        int size = 0; int bi[CAND_K]; double bc[CAND_K];
        for (auto& [d, j] : h) {
          double c = kl.cost(v, i, j);
          if (!std::isfinite(c)) continue;
          if (size == CAND_K && c >= bc[CAND_K - 1]) continue;
          // Capped pairs leave the candidate list, so a splat keeps its cheapest merges the cap allows.
          if (p.cage) { const int m[2] = {(int)i, j}; if (!cap_ok(m, 2)) { capped[i] = 1; blocked_pairs[i]++; continue; } }
          int at = size < CAND_K ? size : CAND_K - 1;
          while (at > 0 && bc[at - 1] > c) { bc[at] = bc[at - 1]; bi[at] = bi[at - 1]; at--; }
          bc[at] = c; bi[at] = j; size = std::min(size + 1, CAND_K);
        }
        for (int s = 0; s < size; s++) { cand_idx[i * CAND_K + s] = bi[s]; cand_cost[i * CAND_K + s] = (float)bc[s]; }
      }
    });
    double t_cost = now_seconds() - t0;

    // decimate-uniform/select.ts: bucketed greedy disjoint matching, then the chain closure at caps 3 and 4.
    const size_t target = std::max(p.count, N - N / 2);
    const size_t needed = N - target;
    const size_t E = N * CAND_K;
    double lo = INFINITY, hi = -INFINITY;
    for (size_t e = 0; e < E; e++) if (std::isfinite(cand_cost[e])) { lo = std::min(lo, (double)cand_cost[e]); hi = std::max(hi, (double)cand_cost[e]); }
    const double span = hi > lo ? hi - lo : 1;
    auto bucket = [&](double c) { return std::min(SELECT_BUCKETS - 1, (int)std::floor((c - lo) / span * SELECT_BUCKETS)); };
    std::vector<size_t> counts(SELECT_BUCKETS + 1, 0);
    for (size_t e = 0; e < E; e++) if (std::isfinite(cand_cost[e])) counts[bucket(cand_cost[e]) + 1]++;
    for (int b = 0; b < SELECT_BUCKETS; b++) counts[b + 1] += counts[b];
    std::vector<uint32_t> order(counts[SELECT_BUCKETS]);
    { std::vector<size_t> cursor(counts.begin(), counts.end() - 1); for (size_t e = 0; e < E; e++) if (std::isfinite(cand_cost[e])) order[cursor[bucket(cand_cost[e])]++] = (uint32_t)e; }

    std::vector<int> group_of(N, -1);
    std::vector<std::vector<int>> groups;
    size_t removed = 0, blocked = 0;
    for (size_t b : blocked_pairs) blocked += b;
    for (size_t t = 0; t < order.size() && removed < needed; t++) {
      const uint32_t e = order[t];
      const int j = cand_idx[e], i = (int)(e / CAND_K);
      if (j < 0 || group_of[i] != -1 || group_of[j] != -1) continue;
      std::vector<int> m = {i, j};
      group_of[i] = group_of[j] = (int)groups.size(); groups.push_back(std::move(m)); removed++;
    }
    for (size_t gcap : {3, 4}) {
      if (removed >= needed) break;
      for (size_t t = 0; t < order.size() && removed < needed; t++) {
        const uint32_t e = order[t];
        const int j = cand_idx[e], i = (int)(e / CAND_K);
        if (j < 0 || group_of[i] != -1) continue;
        const int g = group_of[j];
        if (g == -1 || groups[g].size() >= gcap) continue;
        std::vector<int> m = groups[g]; m.push_back(i);
        if (p.cage && !cap_ok(m.data(), (int)m.size())) { capped[i] = 1; blocked++; continue; }
        group_of[i] = g; groups[g] = std::move(m); removed++;
      }
    }
    size_t held = 0;
    for (size_t i = 0; i < N; i++) held += capped[i] && group_of[i] < 0;
    st.cap_blocked += blocked; st.cap_held = held;
    if (removed == 0) fail("lod-merge: no valid merges at %zu splats (target %zu)%s", N, p.count, p.cage ? "; the cage cap blocks every candidate" : "");
    if (removed < needed && (double)removed / N < MIN_ITERATION_PROGRESS)
      fail("lod-merge: stalled at %zu splats (target %zu): a generation removed only %zu (%.3f%%)%s", N, p.count, removed, 100.0 * removed / N,
           p.cage ? format("; the cage cap ruled out %zu merges in it (raise --max-extent-faces)", blocked).c_str() : "");

    // Survivors keep their place; a group is emitted at its smallest member's place.
    SplatCloud next; next.has_labels = cur.has_labels; next.has_scales = true;
    next.resize(N - removed, cur.sh_degree);
    const int K3 = v.K3;
    size_t o = 0;
    for (size_t i = 0; i < N; i++) {
      const int g = group_of[i];
      if (g < 0) {
        for (int a = 0; a < 3; a++) { next.pos[o * 3 + a] = cur.pos[i * 3 + a]; next.log_scale[o * 3 + a] = cur.log_scale[i * 3 + a]; }
        for (int a = 0; a < 4; a++) next.quat[o * 4 + a] = cur.quat[i * 4 + a];
        next.opacity[o] = cur.opacity[i];
        std::copy(&cur.sh[i * K3], &cur.sh[i * K3] + K3, &next.sh[o * K3]);
        if (cur.has_labels) { next.labels[o * 2] = cur.labels[i * 2]; next.labels[o * 2 + 1] = cur.labels[i * 2 + 1]; }
        o++;
      } else if ((int)i == *std::min_element(groups[g].begin(), groups[g].end())) {
        lod_merge_group(cur, groups[g].data(), (int)groups[g].size(), next, o++);
      }
    }
    size_t sizes[5] = {0, 0, 0, 0, 0}; for (auto& g : groups) sizes[std::min<size_t>(g.size(), 4)]++;
    log_info("lod-merge generation %d: %zu -> %zu splats (%zu groups: %zu pairs, %zu of 3, %zu of 4)%s; costs %.2fs, total %.2fs",
             st.generations, N, next.n, groups.size(), sizes[2], sizes[3], sizes[4], p.cage ? format("; the cage cap ruled out %zu merges and kept %zu splats whole", blocked, held).c_str() : "",
             t_cost, now_seconds() - t0);
    cur = std::move(next);
  }
  if (stats) *stats = st;
  return cur;
}

}  // namespace b2c
