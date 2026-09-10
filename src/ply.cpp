#include "ply.h"
#include "util/log.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <map>
#include <functional>

namespace b2c {

const char* const EVIDENCE_FIELDS[7] = {"ev_w_in", "ev_w_all", "ev_err", "ev_views", "ev_dir_0", "ev_dir_1", "ev_dir_2"};

void SplatCloud::resize(size_t count, int degree) {
  n = count; sh_degree = degree;
  pos.assign(n * 3, 0.f); log_scale.assign(n * 3, 0.f); quat.assign(n * 4, 0.f); opacity.assign(n, 0.f);
  sh.assign(n * K() * 3, 0.f);
  for (size_t i = 0; i < n; i++) quat[i * 4] = 1.f;
  if (has_evidence) evidence.assign(n * 7, 0.f);
}

namespace {
enum class PType { F32, F64, U8, I8, U16, I16, U32, I32 };
struct Prop { std::string name; PType type; };
size_t psize(PType t) {
  switch (t) { case PType::F32: case PType::U32: case PType::I32: return 4; case PType::F64: return 8; case PType::U8: case PType::I8: return 1; default: return 2; }
}
PType parse_type(const std::string& s) {
  if (s == "float" || s == "float32") return PType::F32;
  if (s == "double" || s == "float64") return PType::F64;
  if (s == "uchar" || s == "uint8") return PType::U8;
  if (s == "char" || s == "int8") return PType::I8;
  if (s == "ushort" || s == "uint16") return PType::U16;
  if (s == "short" || s == "int16") return PType::I16;
  if (s == "uint" || s == "uint32") return PType::U32;
  if (s == "int" || s == "int32") return PType::I32;
  fail("ply: unsupported property type '%s'", s.c_str());
}
double read_val(const unsigned char* p, PType t) {
  switch (t) {
    case PType::F32: { float v; memcpy(&v, p, 4); return v; }
    case PType::F64: { double v; memcpy(&v, p, 8); return v; }
    case PType::U8: return *p;
    case PType::I8: return (signed char)*p;
    case PType::U16: { uint16_t v; memcpy(&v, p, 2); return v; }
    case PType::I16: { int16_t v; memcpy(&v, p, 2); return v; }
    case PType::U32: { uint32_t v; memcpy(&v, p, 4); return v; }
    case PType::I32: { int32_t v; memcpy(&v, p, 4); return v; }
  }
  return 0;
}
}  // namespace

bool ply_is_vertex_only(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::string line;
  std::getline(f, line);
  if (line.rfind("ply", 0) != 0) return false;
  bool vertices = false;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::istringstream ss(line);
    std::string tok; ss >> tok;
    if (tok == "element") {
      std::string name; size_t cnt = 0; ss >> name >> cnt;
      if (name == "vertex") vertices = cnt > 0;
      else if (cnt > 0) return false;
    } else if (tok == "end_header") break;
  }
  return vertices;
}

SplatCloud read_ply(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) fail("failed to open ply '%s'", path.c_str());
  std::string line;
  std::getline(f, line);
  if (line.rfind("ply", 0) != 0) fail("'%s' is not a ply file", path.c_str());
  bool binary = false, big_endian = false;
  size_t n_vertex = 0;
  std::vector<Prop> props;
  bool in_vertex = false;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::istringstream ss(line);
    std::string tok; ss >> tok;
    if (tok == "format") { std::string fmt; ss >> fmt; binary = fmt != "ascii"; big_endian = fmt == "binary_big_endian"; }
    else if (tok == "element") { std::string name; size_t cnt; ss >> name >> cnt; in_vertex = name == "vertex"; if (in_vertex) n_vertex = cnt; else if (cnt > 0) fail("ply: unsupported non-vertex element '%s'", name.c_str()); }
    else if (tok == "property" && in_vertex) {
      std::string type; ss >> type;
      if (type == "list") fail("ply: list properties are not supported on vertices");
      Prop p; p.type = parse_type(type); ss >> p.name; props.push_back(p);
    } else if (tok == "end_header") break;
  }
  if (big_endian) fail("ply: big-endian files are not supported");
  std::map<std::string, size_t> col;
  for (size_t i = 0; i < props.size(); i++) col[props[i].name] = i;
  auto has = [&](const char* n) { return col.count(n) > 0; };

  // Read all rows into a column table of doubles.
  size_t stride = 0; for (auto& p : props) stride += psize(p.type);
  std::vector<double> table(n_vertex * props.size());
  if (binary) {
    std::vector<unsigned char> buf(stride * 4096);
    size_t done = 0;
    while (done < n_vertex) {
      size_t chunk = std::min<size_t>(4096, n_vertex - done);
      f.read((char*)buf.data(), chunk * stride);
      if ((size_t)f.gcount() != chunk * stride) fail("ply: truncated file '%s' (got %zu of %zu vertices)", path.c_str(), done + f.gcount() / stride, n_vertex);
      for (size_t r = 0; r < chunk; r++) {
        const unsigned char* p = buf.data() + r * stride;
        for (size_t c = 0; c < props.size(); c++) { table[(done + r) * props.size() + c] = read_val(p, props[c].type); p += psize(props[c].type); }
      }
      done += chunk;
    }
  } else {
    for (size_t r = 0; r < n_vertex; r++)
      for (size_t c = 0; c < props.size(); c++) { double v; if (!(f >> v)) fail("ply: truncated ascii file '%s'", path.c_str()); table[r * props.size() + c] = v; }
  }
  auto at = [&](size_t r, const char* name) { return table[r * props.size() + col.at(name)]; };

  // SH degree from f_rest count.
  int n_rest = 0; while (has(format("f_rest_%d", n_rest).c_str())) n_rest++;
  int K = 1 + n_rest / 3;
  int degree = (int)std::round(std::sqrt((double)K)) - 1;
  if (sh_coeffs_for_degree(degree) != K) fail("ply: %d f_rest properties is not a valid SH layout", n_rest);

  SplatCloud c;
  c.has_evidence = has("ev_w_in");
  c.resize(n_vertex, degree);
  c.has_scales = has("scale_0");
  bool has_rgb = has("red") || has("r");
  const char* rn = has("red") ? "red" : "r"; const char* gn = has("green") ? "green" : "g"; const char* bn = has("blue") ? "blue" : "b";
  double rgb_scale = 1.0;
  if (has_rgb) { PType t = props[col[rn]].type; rgb_scale = t == PType::U8 ? 1.0 / 255.0 : t == PType::U16 ? 1.0 / 65535.0 : 1.0; }
  if (!has("x") || !has("y") || !has("z")) fail("ply: missing x/y/z");
  for (size_t i = 0; i < n_vertex; i++) {
    c.pos[i * 3] = (float)at(i, "x"); c.pos[i * 3 + 1] = (float)at(i, "y"); c.pos[i * 3 + 2] = (float)at(i, "z");
    if (c.has_scales) for (int k = 0; k < 3; k++) c.log_scale[i * 3 + k] = (float)at(i, format("scale_%d", k).c_str());
    if (has("rot_0")) for (int k = 0; k < 4; k++) c.quat[i * 4 + k] = (float)at(i, format("rot_%d", k).c_str());
    if (has("opacity")) c.opacity[i] = (float)at(i, "opacity");
    if (has("f_dc_0")) {
      for (int ch = 0; ch < 3; ch++) c.sh[(i * K) * 3 + ch] = (float)at(i, format("f_dc_%d", ch).c_str());
    } else if (has_rgb) {
      c.sh[(i * K) * 3 + 0] = (float)((at(i, rn) * rgb_scale - 0.5) / SH_C0);
      c.sh[(i * K) * 3 + 1] = (float)((at(i, gn) * rgb_scale - 0.5) / SH_C0);
      c.sh[(i * K) * 3 + 2] = (float)((at(i, bn) * rgb_scale - 0.5) / SH_C0);
    }
    // f_rest is channel-major: all R rest, then G, then B.
    for (int ch = 0; ch < 3; ch++)
      for (int k = 1; k < K; k++)
        c.sh[(i * K + k) * 3 + ch] = (float)at(i, format("f_rest_%d", ch * (K - 1) + (k - 1)).c_str());
    if (c.has_evidence) for (int k = 0; k < 7; k++) c.evidence[i * 7 + k] = (float)at(i, EVIDENCE_FIELDS[k]);
  }
  return c;
}

void write_ply(const std::string& path, const SplatCloud& c, const std::vector<std::string>& comments) {
  std::ofstream f(path, std::ios::binary);
  if (!f) fail("failed to open '%s' for writing", path.c_str());
  int K = c.K();
  std::string h = "ply\nformat binary_little_endian 1.0\n";
  for (auto& cm : comments) h += "comment " + cm + "\n";
  h += format("element vertex %zu\n", c.n);
  const char* base[] = {"x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity", "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"};
  for (auto b : base) h += format("property float %s\n", b);
  for (int k = 0; k < 3 * (K - 1); k++) h += format("property float f_rest_%d\n", k);
  if (c.has_evidence) for (auto e : EVIDENCE_FIELDS) h += format("property float %s\n", e);
  h += "end_header\n";
  f.write(h.data(), h.size());
  size_t row_len = 14 + 3 * (K - 1) + (c.has_evidence ? 7 : 0);
  std::vector<float> row(row_len);
  std::vector<float> buf; buf.reserve(row_len * 4096);
  for (size_t i = 0; i < c.n; i++) {
    float* r = row.data();
    r[0] = c.pos[i * 3]; r[1] = c.pos[i * 3 + 1]; r[2] = c.pos[i * 3 + 2];
    r[3] = c.log_scale[i * 3]; r[4] = c.log_scale[i * 3 + 1]; r[5] = c.log_scale[i * 3 + 2];
    r[6] = c.opacity[i];
    float qw = c.quat[i * 4], qx = c.quat[i * 4 + 1], qy = c.quat[i * 4 + 2], qz = c.quat[i * 4 + 3];
    float qn = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz); if (!(qn > 1e-32f)) { qn = 1; qw = 1; qx = qy = qz = 0; }
    r[7] = qw / qn; r[8] = qx / qn; r[9] = qy / qn; r[10] = qz / qn;
    r[11] = c.sh[(i * K) * 3]; r[12] = c.sh[(i * K) * 3 + 1]; r[13] = c.sh[(i * K) * 3 + 2];
    size_t o = 14;
    for (int ch = 0; ch < 3; ch++) for (int k = 1; k < K; k++) r[o++] = c.sh[(i * K + k) * 3 + ch];
    if (c.has_evidence) for (int k = 0; k < 7; k++) r[o++] = c.evidence[i * 7 + k];
    buf.insert(buf.end(), row.begin(), row.end());
    if (buf.size() >= row_len * 4096) { f.write((const char*)buf.data(), buf.size() * 4); buf.clear(); }
  }
  if (!buf.empty()) f.write((const char*)buf.data(), buf.size() * 4);
  if (!f) fail("failed writing '%s'", path.c_str());
}

}  // namespace b2c
