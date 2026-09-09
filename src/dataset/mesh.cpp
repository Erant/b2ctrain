#include "dataset/mesh.h"
#include "util/log.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>

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
  for (auto& e : elems) {
    bool is_v = e.name == "vertex", is_f = e.name == "face";
    int ix = -1, iy = -1, iz = -1, ilist = -1;
    for (size_t k = 0; k < e.props.size(); k++) {
      if (e.props[k].name == "x") ix = (int)k; else if (e.props[k].name == "y") iy = (int)k; else if (e.props[k].name == "z") iz = (int)k;
      if (e.props[k].list && (e.props[k].name == "vertex_indices" || e.props[k].name == "vertex_index")) ilist = (int)k;
    }
    if (is_v && (ix < 0 || iy < 0 || iz < 0)) fail("mesh '%s': vertex element without x/y/z", path.c_str());
    if (is_f && ilist < 0) fail("mesh '%s': face element without vertex_indices", path.c_str());
    if (is_v) m.v.reserve(e.count * 3);
    std::vector<double> vals(e.props.size());
    std::vector<uint32_t> idx;
    std::vector<unsigned char> buf;
    for (size_t n = 0; n < e.count; n++) {
      idx.clear();
      if (binary) {
        for (size_t k = 0; k < e.props.size(); k++) {
          const Prop& p = e.props[k];
          if (p.list) {
            buf.resize(psize(p.count_type)); f.read((char*)buf.data(), buf.size());
            size_t cnt = (size_t)rd(buf.data(), p.count_type);
            buf.resize(psize(p.type) * cnt); f.read((char*)buf.data(), buf.size());
            if ((int)k == ilist) for (size_t j = 0; j < cnt; j++) idx.push_back((uint32_t)rd(buf.data() + j * psize(p.type), p.type));
          } else { buf.resize(psize(p.type)); f.read((char*)buf.data(), buf.size()); vals[k] = rd(buf.data(), p.type); }
        }
        if (!f) fail("mesh '%s': truncated ply data", path.c_str());
      } else {
        if (!std::getline(f, line)) fail("mesh '%s': truncated ply data", path.c_str());
        std::istringstream ss(line);
        for (size_t k = 0; k < e.props.size(); k++) {
          const Prop& p = e.props[k];
          if (p.list) { size_t cnt = 0; ss >> cnt; for (size_t j = 0; j < cnt; j++) { uint32_t v = 0; ss >> v; if ((int)k == ilist) idx.push_back(v); } }
          else ss >> vals[k];
        }
      }
      if (is_v) { m.v.push_back((float)vals[ix]); m.v.push_back((float)vals[iy]); m.v.push_back((float)vals[iz]); }
      if (is_f) push_face(idx);
    }
  }
  return m;
}

TriMesh read_obj_mesh(const std::string& path) {
  std::ifstream f(path);
  if (!f) fail("failed to open mesh '%s'", path.c_str());
  TriMesh m; std::string line;
  while (std::getline(f, line)) {
    if (line.size() < 2) continue;
    if (line[0] == 'v' && line[1] == ' ') { std::istringstream ss(line.substr(2)); float x, y, z; ss >> x >> y >> z; m.v.push_back(x); m.v.push_back(y); m.v.push_back(z); }
    else if (line[0] == 'f' && line[1] == ' ') {
      std::istringstream ss(line.substr(2)); std::string tok; std::vector<uint32_t> idx;
      while (ss >> tok) { long v = strtol(tok.c_str(), nullptr, 10); if (v < 0) v = (long)(m.v.size() / 3) + v + 1; idx.push_back((uint32_t)(v - 1)); }
      for (size_t k = 2; k < idx.size(); k++) { m.f.push_back(idx[0]); m.f.push_back(idx[k - 1]); m.f.push_back(idx[k]); }
    }
  }
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

}  // namespace b2c
