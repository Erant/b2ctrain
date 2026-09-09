#include "reference.h"
#include <cmath>
#include <algorithm>
#include <array>

using namespace b2c;
namespace {
struct V3 { double x, y, z; };
V3 mul(const double* R, V3 v) { return {R[0] * v.x + R[1] * v.y + R[2] * v.z, R[3] * v.x + R[4] * v.y + R[5] * v.z, R[6] * v.x + R[7] * v.y + R[8] * v.z}; }
void quat_mat(double w, double x, double y, double z, double* R) {
  double n = std::sqrt(w * w + x * x + y * y + z * z); w /= n; x /= n; y /= n; z /= n;
  R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - z * w); R[2] = 2 * (x * z + y * w);
  R[3] = 2 * (x * y + z * w); R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - x * w);
  R[6] = 2 * (x * z - y * w); R[7] = 2 * (y * z + x * w); R[8] = 1 - 2 * (x * x + y * y);
}
std::vector<double> sh_basis(int deg, V3 v) {
  std::vector<double> b((deg + 1) * (deg + 1), 0.0);
  double x = v.x, y = v.y, z = v.z;
  b[0] = 0.28209479177387814;
  if (deg >= 1) { b[1] = -0.4886025 * y; b[2] = 0.4886025 * z; b[3] = -0.4886025 * x; }
  if (deg >= 2) {
    double z2 = z * z, f0b = -1.0925485 * z, f1a = 0.54627424, fc1 = x * x - y * y, fs1 = 2 * x * y;
    b[4] = f1a * fs1; b[5] = f0b * y; b[6] = 0.9461747 * z2 - 0.31539157; b[7] = f0b * x; b[8] = f1a * fc1;
    if (deg >= 3) {
      double f0c = -2.285229 * z2 + 0.4570458, f1b = 1.4453057 * z, f2a = -0.5900436, fc2 = x * fc1 - y * fs1, fs2 = x * fs1 + y * fc1;
      b[9] = f2a * fs2; b[10] = f1b * fs1; b[11] = f0c * y; b[12] = z * (1.8658817 * z2 - 1.119529); b[13] = f0c * x; b[14] = f1b * fc1; b[15] = f2a * fc2;
    }
  }
  return b;
}
struct Proj { bool ok = false; double mx, my, c00, c01, c11, opac, depth, r, g, b, fx, fy, fz; };
}  // namespace

double reference_loss(const SplatCloud& c, const RefParams& p) {
  const Camera& cam = p.cam;
  double R[9]; for (int i = 0; i < 9; i++) R[i] = cam.R[i];
  double lim_pos_x = (1.15 * p.W - cam.cx) / cam.fx, lim_neg_x = (-0.15 * p.W - cam.cx) / cam.fx;
  double lim_pos_y = (1.15 * p.H - cam.cy) / cam.fy, lim_neg_y = (-0.15 * p.H - cam.cy) / cam.fy;
  int K = c.K();
  std::vector<Proj> pr(c.n);
  for (size_t i = 0; i < c.n; i++) {
    V3 mean{c.pos[i * 3], c.pos[i * 3 + 1], c.pos[i * 3 + 2]};
    V3 mc = mul(R, mean); mc.x += cam.t[0]; mc.y += cam.t[1]; mc.z += cam.t[2];
    if (!(mc.z >= 0.01)) continue;
    double s[3]; for (int k = 0; k < 3; k++) s[k] = std::exp(c.log_scale[i * 3 + k]);
    double Rq[9]; quat_mat(c.quat[i * 4], c.quat[i * 4 + 1], c.quat[i * 4 + 2], c.quat[i * 4 + 3], Rq);
    double M[9];
    for (int r = 0; r < 3; r++) for (int col = 0; col < 3; col++) { double v = 0; for (int k = 0; k < 3; k++) v += R[r * 3 + k] * Rq[k * 3 + col]; M[r * 3 + col] = v * s[col]; }
    double inv_z = 1.0 / mc.z, rx = mc.x * inv_z, ry = mc.y * inv_z;
    double cxr = std::min(std::max(rx, lim_neg_x), lim_pos_x), cyr = std::min(std::max(ry, lim_neg_y), lim_pos_y);
    double J[6] = {cam.fx * inv_z, 0, -cam.fx * inv_z * cxr, 0, cam.fy * inv_z, -cam.fy * inv_z * cyr};
    double V[6]; for (int r = 0; r < 2; r++) for (int col = 0; col < 3; col++) V[r * 3 + col] = J[r * 3] * M[col] + J[r * 3 + 1] * M[3 + col] + J[r * 3 + 2] * M[6 + col];
    double c00 = V[0] * V[0] + V[1] * V[1] + V[2] * V[2] + 0.3, c01 = V[0] * V[3] + V[1] * V[4] + V[2] * V[5], c11 = V[3] * V[3] + V[4] * V[4] + V[5] * V[5] + 0.3;
    double det = c00 * c11 - c01 * c01; if (!(det > 0)) continue;
    Proj q; q.c00 = c11 / det; q.c01 = -c01 / det; q.c11 = c00 / det;
    q.opac = 1.0 / (1.0 + std::exp(-c.opacity[i]));
    if (!(q.opac >= 1.0 / 255.0)) continue;
    q.mx = cam.fx * rx + cam.cx; q.my = cam.fy * ry + cam.cy; q.depth = mc.z;
    double power = std::log(q.opac * 255.0), detc = q.c00 * q.c11 - q.c01 * q.c01;
    double ex = std::sqrt(2 * power * q.c11 / detc), ey = std::sqrt(2 * power * q.c00 / detc);
    if (!(q.mx + ex > 0 && q.mx - ex < p.W && q.my + ey > 0 && q.my - ey < p.H)) continue;
    V3 v{mean.x - cam.pos[0], mean.y - cam.pos[1], mean.z - cam.pos[2]}; double vl = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); v.x /= vl; v.y /= vl; v.z /= vl;
    auto b = sh_basis(c.sh_degree, v);
    double col[3] = {0, 0, 0};
    for (int k = 0; k < K; k++) for (int ch = 0; ch < 3; ch++) col[ch] += b[k] * c.sh[(i * K + k) * 3 + ch];
    q.r = std::min(std::max(col[0] + 0.5, -100.0), 100.0); q.g = std::min(std::max(col[1] + 0.5, -100.0), 100.0); q.b = std::min(std::max(col[2] + 0.5, -100.0), 100.0);
    if (p.normals) {
      const float* ls = &c.log_scale[i * 3];
      int axis = (ls[0] <= ls[1] && ls[0] <= ls[2]) ? 0 : (ls[1] <= ls[2] ? 1 : 2);
      V3 a{Rq[axis], Rq[3 + axis], Rq[6 + axis]}; double al = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
      V3 u{a.x / al, a.y / al, a.z / al};
      double face = ((cam.pos[0] - mean.x) * u.x + (cam.pos[1] - mean.y) * u.y + (cam.pos[2] - mean.z) * u.z) >= 0 ? 1.0 : -1.0;
      V3 n = mul(R, V3{u.x * face, u.y * face, u.z * face});
      q.fx = n.x; q.fy = n.y; q.fz = n.z;
    }
    q.ok = true; pr[i] = q;
  }
  std::vector<int> order; for (size_t i = 0; i < c.n; i++) if (pr[i].ok) order.push_back((int)i);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return pr[a].depth < pr[b].depth; });
  // Rasterise.
  int W = p.W, H = p.H;
  std::vector<double> img((size_t)W * H * 4, 0.0), feat((size_t)W * H * 3, 0.0);
  double hollow = 0.0;
  for (int py = 0; py < H; py++) for (int px = 0; px < W; px++) {
    double T = 1, r = 0, g = 0, b = 0, f0 = 0, f1 = 0, f2 = 0;
    double pcx = px + 0.5, pcy = py + 0.5;
    for (int i : order) {
      const Proj& q = pr[i];
      double dx = pcx - q.mx, dy = pcy - q.my;
      double sigma = 0.5 * (q.c00 * dx * dx + q.c11 * dy * dy) + q.c01 * dx * dy;
      double alpha = std::min(0.999, q.opac * std::exp(-sigma));
      if (!(sigma >= 0 && alpha >= 1.0 / 255.0)) continue;
      double nT = T * (1 - alpha);
      if (nT <= 1e-4) break;
      double vis = alpha * T;
      r += std::max(q.r, 0.0) * vis; g += std::max(q.g, 0.0) * vis; b += std::max(q.b, 0.0) * vis;
      f0 += q.fx * vis; f1 += q.fy * vis; f2 += q.fz * vis;
      if (p.hollow_z) { double zr = (*p.hollow_z)[(size_t)py * W + px]; double d = (q.depth - zr - p.hollow_margin) / std::max(p.hollow_margin, 1e-4); if (std::isfinite(d)) hollow += vis * std::min(std::max(d, 0.0), 1.0); }
      T = nT;
    }
    size_t o = (size_t)py * W + px;
    img[o * 4] = r + T * p.bg[0]; img[o * 4 + 1] = g + T * p.bg[1]; img[o * 4 + 2] = b + T * p.bg[2]; img[o * 4 + 3] = 1 - T;
    feat[o * 3] = f0; feat[o * 3 + 1] = f1; feat[o * 3 + 2] = f2;
  }
  // Loss.
  double gauss[11]; { double sum = 0; for (int i = 0; i < 11; i++) { double x = i - 5.0; gauss[i] = std::exp(-x * x / (2 * 1.5 * 1.5)); sum += gauss[i]; } for (auto& w : gauss) w /= sum; }
  auto gtc = [&](int pix, int ch) { return ((*p.gt)[pix] >> (8 * ch) & 0xff) / 255.0; };
  double loss = 0;
  const double norm = p.scale / (3.0 * W * H);
  for (int ch = 0; ch < 3; ch++) {
    std::vector<double> X((size_t)W * H), Y((size_t)W * H);
    for (int i = 0; i < W * H; i++) { X[i] = img[(size_t)i * 4 + ch]; Y[i] = gtc(i, ch) + (p.composite ? (1 - gtc(i, 3)) * p.bg[ch] : 0.0); }
    auto blur = [&](const std::vector<double>& A) {
      std::vector<double> t((size_t)W * H, 0.0), o((size_t)W * H, 0.0);
      for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) { double s = 0; for (int d = -5; d <= 5; d++) { int xx = x + d; if (xx >= 0 && xx < W) s += gauss[d + 5] * A[(size_t)y * W + xx]; } t[(size_t)y * W + x] = s; }
      for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) { double s = 0; for (int d = -5; d <= 5; d++) { int yy = y + d; if (yy >= 0 && yy < H) s += gauss[d + 5] * t[(size_t)yy * W + x]; } o[(size_t)y * W + x] = s; }
      return o;
    };
    std::vector<double> X2(X.size()), Y2(X.size()), XY(X.size());
    for (size_t i = 0; i < X.size(); i++) { X2[i] = X[i] * X[i]; Y2[i] = Y[i] * Y[i]; XY[i] = X[i] * Y[i]; }
    auto mu1 = blur(X), mu2 = blur(Y), ex2 = blur(X2), ey2 = blur(Y2), exy = blur(XY);
    for (int i = 0; i < W * H; i++) {
      double s1 = std::max(0.0, ex2[i] - mu1[i] * mu1[i]), s2 = std::max(0.0, ey2[i] - mu2[i] * mu2[i]), s12 = exy[i] - mu1[i] * mu2[i];
      double ssim = ((2 * mu1[i] * mu2[i] + 1e-4) * (2 * s12 + 9e-4)) / ((mu1[i] * mu1[i] + mu2[i] * mu2[i] + 1e-4) * (s1 + s2 + 9e-4));
      ssim = std::min(std::max(ssim, -1.0), 1.0);
      double wp = p.wts ? (*p.wts)[i] / 255.0 : 1.0;
      double M = wp * (p.mask ? gtc(i, 3) : 1.0) * norm;
      loss += M * (p.l1_w * std::abs(X[i] - Y[i]) + p.ssim_w * ssim);
    }
  }
  if (p.alpha_lane) for (int i = 0; i < W * H; i++) { double wp = p.wts ? (*p.wts)[i] / 255.0 : 1.0; loss += wp * p.match_alpha_weight * p.scale / (double)(W * H) * std::abs(img[(size_t)i * 4 + 3] - gtc(i, 3)); }
  if (p.normals && p.normal_scale > 0) for (int i = 0; i < W * H; i++) {
    uint32_t g = (*p.gtn)[i];
    if (((g >> 24) & 0xff) <= 127) continue;
    double wp = p.wts ? (*p.wts)[i] / 255.0 : 1.0;
    double gx = (g & 0xff) * (2.0 / 255.0) - 1, gy = 1 - ((g >> 8) & 0xff) * (2.0 / 255.0), gz = 1 - ((g >> 16) & 0xff) * (2.0 / 255.0);
    double px = feat[(size_t)i * 3], py = feat[(size_t)i * 3 + 1], pz = feat[(size_t)i * 3 + 2];
    double pl = std::max(std::sqrt(px * px + py * py + pz * pz), 1e-6), gl = std::max(std::sqrt(gx * gx + gy * gy + gz * gz), 1e-6);
    loss += p.normal_scale * wp * (std::abs(px - gx) + std::abs(py - gy) + std::abs(pz - gz) + 1 - (px * gx + py * gy + pz * gz) / (pl * gl));
  }
  loss += p.hollow_lam * hollow;
  return loss;
}
