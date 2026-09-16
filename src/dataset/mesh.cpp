#include "dataset/mesh.h"
#include "util/log.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cstdio>

namespace b2c {

namespace {
enum class PT { F32, F64, U8, I8, U16, I16, U32, I32 };
size_t psize(PT t) { switch (t) { case PT::F32: case PT::U32: case PT::I32: return 4; case PT::F64: return 8; case PT::U8: case PT::I8: return 1; default: return 2; } }
PT ptype(const std::string& s, const std::string& path) {
  if (s == "float" || s == "float32") return PT::F32;
  if (s == "double" || s == "float64") return PT::F64;
  if (s == "uchar" || s == "uint8") return PT::U8;
  if (s == "char" || s == "int8") return PT::I8;
  if (s == "ushort" || s == "uint16") return PT::U16;
  if (s == "short" || s == "int16") return PT::I16;
  if (s == "uint" || s == "uint32") return PT::U32;
  if (s == "int" || s == "int32") return PT::I32;
  fail("mesh '%s': unsupported ply property type '%s'", path.c_str(), s.c_str());
}
double rd(const unsigned char* p, PT t) {
  switch (t) {
    case PT::F32: { float v; memcpy(&v, p, 4); return v; }
    case PT::F64: { double v; memcpy(&v, p, 8); return v; }
    case PT::U8: return *p;
    case PT::I8: return (signed char)*p;
    case PT::U16: { uint16_t v; memcpy(&v, p, 2); return v; }
    case PT::I16: { int16_t v; memcpy(&v, p, 2); return v; }
    case PT::U32: { uint32_t v; memcpy(&v, p, 4); return v; }
    case PT::I32: { int32_t v; memcpy(&v, p, 4); return v; }
  }
  return 0;
}
struct Prop { std::string name; PT type = PT::F32; bool list = false; PT count_type = PT::U8; };
struct Elem { std::string name; size_t count = 0; std::vector<Prop> props; };

TriMesh read_ply_mesh(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) fail("failed to open mesh '%s'", path.c_str());
  std::string line; std::getline(f, line);
  if (line.rfind("ply", 0) != 0) fail("'%s' is not a ply file", path.c_str());
  bool binary = false;
  std::vector<Elem> elems;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::istringstream ss(line); std::string kw; ss >> kw;
    if (kw == "format") { std::string fmt; ss >> fmt; if (fmt == "binary_little_endian") binary = true; else if (fmt == "ascii") binary = false; else fail("mesh '%s': unsupported ply format '%s'", path.c_str(), fmt.c_str()); }
    else if (kw == "element") { Elem e; ss >> e.name >> e.count; elems.push_back(e); }
    else if (kw == "property") {
      if (elems.empty()) fail("mesh '%s': property before element", path.c_str());
      std::string t; ss >> t; Prop p;
      if (t == "list") { std::string ct, vt; ss >> ct >> vt >> p.name; p.list = true; p.count_type = ptype(ct, path); p.type = ptype(vt, path); }
      else { p.type = ptype(t, path); ss >> p.name; }
      elems.back().props.push_back(p);
    }
    else if (kw == "end_header") break;
  }
  TriMesh m;
  auto push_face = [&](const std::vector<uint32_t>& idx) { for (size_t k = 2; k < idx.size(); k++) { m.f.push_back(idx[0]); m.f.push_back(idx[k - 1]); m.f.push_back(idx[k]); } };
  // The binary body is read in one go and parsed from memory (a per-property stream read is slow on a million vertices).
  std::vector<unsigned char> body; size_t off = 0;
  if (binary) { std::streampos here = f.tellg(); f.seekg(0, std::ios::end); size_t total = (size_t)(f.tellg() - here); f.seekg(here); body.resize(total); f.read((char*)body.data(), total); if ((size_t)f.gcount() != total) fail("mesh '%s': truncated ply data", path.c_str()); }
  auto take = [&](size_t n) { if (off + n > body.size()) fail("mesh '%s': truncated ply data", path.c_str()); const unsigned char* p = body.data() + off; off += n; return p; };
  for (auto& e : elems) {
    bool is_v = e.name == "vertex", is_f = e.name == "face";
    int ix = -1, iy = -1, iz = -1, ilist = -1, ir = -1, ig = -1, ib = -1, inx = -1, iny = -1, inz = -1;
    for (size_t k = 0; k < e.props.size(); k++) {
      const std::string& pn = e.props[k].name;
      if (pn == "x") ix = (int)k; else if (pn == "y") iy = (int)k; else if (pn == "z") iz = (int)k;
      else if (pn == "red" || pn == "r") ir = (int)k; else if (pn == "green" || pn == "g") ig = (int)k; else if (pn == "blue" || pn == "b") ib = (int)k;
      else if (pn == "nx") inx = (int)k; else if (pn == "ny") iny = (int)k; else if (pn == "nz") inz = (int)k;
      if (e.props[k].list && (pn == "vertex_indices" || pn == "vertex_index")) ilist = (int)k;
    }
    bool has_col = is_v && ir >= 0 && ig >= 0 && ib >= 0, has_nrm = is_v && inx >= 0 && iny >= 0 && inz >= 0;
    // Colours as 0..255 whatever the property type (a float colour is taken as 0..1).
    auto col8 = [&](double c, PT t) { double s = (t == PT::F32 || t == PT::F64) ? c * 255.0 : c; return (uint8_t)(s < 0 ? 0 : s > 255 ? 255 : s + 0.5); };
    if (is_v && (ix < 0 || iy < 0 || iz < 0)) fail("mesh '%s': vertex element without x/y/z", path.c_str());
    if (is_f && ilist < 0) fail("mesh '%s': face element without vertex_indices", path.c_str());
    if (is_v) m.v.reserve(e.count * 3);
    std::vector<double> vals(e.props.size());
    std::vector<uint32_t> idx;
    if (is_f) m.f.reserve(e.count * 3);
    for (size_t n = 0; n < e.count; n++) {
      idx.clear();
      if (binary) {
        for (size_t k = 0; k < e.props.size(); k++) {
          const Prop& p = e.props[k];
          if (p.list) {
            size_t cnt = (size_t)rd(take(psize(p.count_type)), p.count_type);
            const unsigned char* d = take(psize(p.type) * cnt);
            if ((int)k == ilist) for (size_t j = 0; j < cnt; j++) idx.push_back((uint32_t)rd(d + j * psize(p.type), p.type));
          } else vals[k] = rd(take(psize(p.type)), p.type);
        }
      } else {
        if (!std::getline(f, line)) fail("mesh '%s': truncated ply data", path.c_str());
        std::istringstream ss(line);
        for (size_t k = 0; k < e.props.size(); k++) {
          const Prop& p = e.props[k];
          if (p.list) { size_t cnt = 0; ss >> cnt; for (size_t j = 0; j < cnt; j++) { uint32_t v = 0; ss >> v; if ((int)k == ilist) idx.push_back(v); } }
          else ss >> vals[k];
        }
      }
      if (is_v) {
        m.v.push_back((float)vals[ix]); m.v.push_back((float)vals[iy]); m.v.push_back((float)vals[iz]);
        if (has_col) { m.col.push_back(col8(vals[ir], e.props[ir].type)); m.col.push_back(col8(vals[ig], e.props[ig].type)); m.col.push_back(col8(vals[ib], e.props[ib].type)); }
        if (has_nrm) { m.nrm.push_back((float)vals[inx]); m.nrm.push_back((float)vals[iny]); m.nrm.push_back((float)vals[inz]); }
      }
      if (is_f) push_face(idx);
    }
  }
  return m;
}

TriMesh read_obj_mesh(const std::string& path) {
  std::ifstream f(path);
  if (!f) fail("failed to open mesh '%s'", path.c_str());
  TriMesh m; std::string line; bool any_vt = false;
  while (std::getline(f, line)) {
    if (line.size() < 2) continue;
    if (line[0] == 'v' && line[1] == ' ') { std::istringstream ss(line.substr(2)); float x, y, z; ss >> x >> y >> z; m.v.push_back(x); m.v.push_back(y); m.v.push_back(z); }
    else if (line[0] == 'v' && line[1] == 't' && line.size() > 2 && line[2] == ' ') { std::istringstream ss(line.substr(3)); float u = 0, w = 0; ss >> u >> w; m.uv.push_back(u); m.uv.push_back(w); }
    else if (line[0] == 'f' && line[1] == ' ') {
      std::istringstream ss(line.substr(2)); std::string tok; std::vector<uint32_t> idx, tidx;
      while (ss >> tok) {
        char* end = nullptr; long v = strtol(tok.c_str(), &end, 10); if (v < 0) v = (long)(m.v.size() / 3) + v + 1; idx.push_back((uint32_t)(v - 1));
        if (*end == '/' && end[1] != '/' && end[1] != '\0') { long t = strtol(end + 1, nullptr, 10); if (t < 0) t = (long)(m.uv.size() / 2) + t + 1; tidx.push_back((uint32_t)(t - 1)); any_vt = true; }
        else tidx.push_back(0);
      }
      for (size_t k = 2; k < idx.size(); k++) {
        m.f.push_back(idx[0]); m.f.push_back(idx[k - 1]); m.f.push_back(idx[k]);
        m.fuv.push_back(tidx[0]); m.fuv.push_back(tidx[k - 1]); m.fuv.push_back(tidx[k]);
      }
    }
  }
  if (!any_vt || m.uv.empty()) { m.uv.clear(); m.fuv.clear(); }
  else for (uint32_t i : m.fuv) if (i >= m.uv.size() / 2) fail("mesh '%s': uv index %u out of range (%zu uvs)", path.c_str(), i, m.uv.size() / 2);
  return m;
}
}  // namespace

TriMesh read_mesh(const std::string& path) {
  std::string ext = path.size() >= 4 ? path.substr(path.size() - 4) : "";
  for (auto& c : ext) c = (char)tolower((unsigned char)c);
  TriMesh m = ext == ".obj" ? read_obj_mesh(path) : read_ply_mesh(path);
  for (uint32_t i : m.f) if (i >= m.nv()) fail("mesh '%s': face index %u out of range (%zu vertices)", path.c_str(), i, m.nv());
  if (m.nf() == 0) fail("mesh '%s' has no triangles", path.c_str());
  return m;
}

void write_ply_mesh(const std::string& path, const TriMesh& m) {
  FILE* fp = fopen(path.c_str(), "wb");
  if (!fp) fail("failed to write '%s'", path.c_str());
  bool nrm = m.has_normals(), col = m.has_colour();
  std::string hdr = "ply\nformat binary_little_endian 1.0\ncomment Created by b2ctrain\nelement vertex " + std::to_string(m.nv()) + "\nproperty float x\nproperty float y\nproperty float z\n";
  if (nrm) hdr += "property float nx\nproperty float ny\nproperty float nz\n";
  if (col) hdr += "property uchar red\nproperty uchar green\nproperty uchar blue\n";
  hdr += "element face " + std::to_string(m.nf()) + "\nproperty list uchar uint vertex_indices\nend_header\n";
  fwrite(hdr.data(), 1, hdr.size(), fp);
  size_t rec = 12 + (nrm ? 12 : 0) + (col ? 3 : 0);
  std::vector<unsigned char> buf(rec * m.nv());
  for (size_t i = 0; i < m.nv(); i++) {
    unsigned char* p = buf.data() + i * rec;
    memcpy(p, &m.v[i * 3], 12); p += 12;
    if (nrm) { memcpy(p, &m.nrm[i * 3], 12); p += 12; }
    if (col) { p[0] = m.col[i * 3]; p[1] = m.col[i * 3 + 1]; p[2] = m.col[i * 3 + 2]; }
  }
  fwrite(buf.data(), 1, buf.size(), fp);
  buf.assign(13 * m.nf(), 0);
  for (size_t i = 0; i < m.nf(); i++) { unsigned char* p = buf.data() + i * 13; p[0] = 3; memcpy(p + 1, &m.f[i * 3], 12); }
  fwrite(buf.data(), 1, buf.size(), fp);
  if (fclose(fp)) fail("failed to write '%s'", path.c_str());
}

void write_obj_mesh(const std::string& path, const TriMesh& m, const std::string& texture) {
  std::string stem = path, dir;
  auto slash = stem.find_last_of('/'); if (slash != std::string::npos) { dir = stem.substr(0, slash + 1); stem = stem.substr(slash + 1); }
  auto dot = stem.rfind('.'); if (dot != std::string::npos) stem = stem.substr(0, dot);
  FILE* fp = fopen(path.c_str(), "w");
  if (!fp) fail("failed to write '%s'", path.c_str());
  std::string out; out.reserve(m.nv() * 40 + m.nf() * 48);
  char line[128];
  out += "mtllib " + stem + ".mtl\n";
  for (size_t i = 0; i < m.nv(); i++) { snprintf(line, sizeof line, "v %.6f %.6f %.6f\n", m.v[i * 3], m.v[i * 3 + 1], m.v[i * 3 + 2]); out += line; }
  for (size_t i = 0; i < m.uv.size() / 2; i++) { snprintf(line, sizeof line, "vt %.6f %.6f\n", m.uv[i * 2], m.uv[i * 2 + 1]); out += line; }
  out += "usemtl tex\n";
  bool has_uv = m.has_uv();
  for (size_t i = 0; i < m.nf(); i++) {
    if (has_uv) snprintf(line, sizeof line, "f %u/%u %u/%u %u/%u\n", m.f[i * 3] + 1, m.fuv[i * 3] + 1, m.f[i * 3 + 1] + 1, m.fuv[i * 3 + 1] + 1, m.f[i * 3 + 2] + 1, m.fuv[i * 3 + 2] + 1);
    else snprintf(line, sizeof line, "f %u %u %u\n", m.f[i * 3] + 1, m.f[i * 3 + 1] + 1, m.f[i * 3 + 2] + 1);
    out += line;
  }
  fwrite(out.data(), 1, out.size(), fp);
  if (fclose(fp)) fail("failed to write '%s'", path.c_str());
  FILE* mp = fopen((dir + stem + ".mtl").c_str(), "w");
  if (!mp) fail("failed to write '%s'", (dir + stem + ".mtl").c_str());
  fprintf(mp, "newmtl tex\nKd 1 1 1\nmap_Kd %s\n", texture.c_str());
  fclose(mp);
}

}  // namespace b2c
