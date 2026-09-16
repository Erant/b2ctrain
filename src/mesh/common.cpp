#include "mesh/common.h"
#include "render/renderer.h"
#include "ply.h"
#include "util/log.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "json.hpp"
#include <fstream>
#include <thread>
#include <atomic>
#include <algorithm>
#include <cstring>
#include <sys/stat.h>

namespace b2c {

CamSet load_cams(const std::string& path) {
  std::ifstream cf(path); if (!cf) fail("failed to open cameras '%s'", path.c_str());
  nlohmann::json j; cf >> j;
  CamSet cs; cs.W = j.at("width").get<int>(); cs.H = j.at("height").get<int>();
  for (auto& c : j.at("cameras")) { cs.cams.push_back(camera_from_json(c, cs.W, cs.H)); cs.names.push_back(c.at("name").get<std::string>()); }
  return cs;
}

std::string file_stem(const std::string& name) {
  std::string s = name; auto slash = s.find_last_of('/'); if (slash != std::string::npos) s = s.substr(slash + 1);
  auto dot = s.rfind('.'); if (dot != std::string::npos) s = s.substr(0, dot);
  return s;
}

bool file_exists(const std::string& path) { struct stat st; return stat(path.c_str(), &st) == 0; }

std::vector<float> read_points(const std::string& path) {
  std::string ext = path.size() >= 4 ? path.substr(path.size() - 4) : "";
  if (ext == ".ply" && ply_is_vertex_only(path)) return read_ply(path).pos;
  return read_mesh(path).v;
}

bool try_load_image8(const std::string& path, int channels, Image8& out) {
  int w = 0, h = 0, c = 0;
  unsigned char* d = stbi_load(path.c_str(), &w, &h, &c, channels);
  if (!d) return false;
  out.W = w; out.H = h; out.C = channels; out.px.assign(d, d + (size_t)w * h * channels);
  stbi_image_free(d);
  return true;
}
Image8 load_image8(const std::string& path, int channels) {
  Image8 im; if (!try_load_image8(path, channels, im)) fail("failed to read image '%s'", path.c_str());
  return im;
}

bool try_load_png16(const std::string& path, int& W, int& H, std::vector<uint16_t>& out) {
  int c = 0;
  unsigned short* d = stbi_load_16(path.c_str(), &W, &H, &c, 1);
  if (!d) return false;
  out.assign(d, d + (size_t)W * H);
  stbi_image_free(d);
  return true;
}

namespace {
uint32_t crc32_of(const unsigned char* d, size_t n, uint32_t c = 0xffffffffu) {
  for (size_t i = 0; i < n; i++) { c ^= d[i]; for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1u))); }
  return c;
}
uint32_t adler32_of(const unsigned char* d, size_t n) { uint32_t a = 1, b = 0; for (size_t i = 0; i < n; i++) { a = (a + d[i]) % 65521u; b = (b + a) % 65521u; } return (b << 16) | a; }
void put32(std::vector<unsigned char>& v, uint32_t x) { v.push_back(x >> 24); v.push_back(x >> 16); v.push_back(x >> 8); v.push_back(x); }
void chunk(std::vector<unsigned char>& out, const char* type, const std::vector<unsigned char>& data) {
  put32(out, (uint32_t)data.size());
  std::vector<unsigned char> td(type, type + 4); td.insert(td.end(), data.begin(), data.end());
  out.insert(out.end(), td.begin(), td.end());
  put32(out, crc32_of(td.data(), td.size()) ^ 0xffffffffu);
}
}  // namespace

// 16-bit greyscale PNG with stored (uncompressed) deflate blocks: stb has no 16-bit writer.
void write_png16(const std::string& path, int W, int H, const uint16_t* px) {
  std::vector<unsigned char> raw; raw.reserve((size_t)H * (1 + 2 * W));
  for (int y = 0; y < H; y++) { raw.push_back(0); for (int x = 0; x < W; x++) { uint16_t v = px[(size_t)y * W + x]; raw.push_back(v >> 8); raw.push_back(v & 0xff); } }
  std::vector<unsigned char> z = {0x78, 0x01};
  for (size_t off = 0; off < raw.size();) {
    size_t n = std::min<size_t>(65535, raw.size() - off); bool last = off + n == raw.size();
    z.push_back(last ? 1 : 0); z.push_back(n & 0xff); z.push_back(n >> 8); z.push_back(~n & 0xff); z.push_back((~n >> 8) & 0xff);
    z.insert(z.end(), raw.begin() + off, raw.begin() + off + n); off += n;
  }
  put32(z, adler32_of(raw.data(), raw.size()));
  std::vector<unsigned char> out = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
  std::vector<unsigned char> ihdr; put32(ihdr, W); put32(ihdr, H); ihdr.push_back(16); ihdr.push_back(0); ihdr.push_back(0); ihdr.push_back(0); ihdr.push_back(0);
  chunk(out, "IHDR", ihdr); chunk(out, "IDAT", z); chunk(out, "IEND", {});
  FILE* f = fopen(path.c_str(), "wb"); if (!f) fail("failed to write '%s'", path.c_str()); fwrite(out.data(), 1, out.size(), f); fclose(f);
}

void write_png8(const std::string& path, int W, int H, int C, const uint8_t* px) {
  if (!stbi_write_png(path.c_str(), W, H, C, px, W * C)) fail("failed to write '%s'", path.c_str());
}

void write_f32(const std::string& path, const float* data, size_t n) {
  FILE* f = fopen(path.c_str(), "wb"); if (!f) fail("failed to write '%s'", path.c_str());
  if (fwrite(data, sizeof(float), n, f) != n) fail("failed to write '%s'", path.c_str());
  fclose(f);
}
std::vector<float> read_f32(const std::string& path, size_t n_expected) {
  FILE* f = fopen(path.c_str(), "rb"); if (!f) fail("failed to open '%s'", path.c_str());
  std::vector<float> v(n_expected);
  size_t got = fread(v.data(), sizeof(float), n_expected, f); fclose(f);
  if (got != n_expected) fail("'%s': expected %zu floats, found %zu", path.c_str(), n_expected, got);
  return v;
}

void parallel_for(size_t n, const std::function<void(size_t)>& fn, unsigned max_threads) {
  unsigned nt = std::max(1u, std::min(std::thread::hardware_concurrency(), max_threads ? max_threads : 32u));
  nt = (unsigned)std::min<size_t>(nt, n);
  if (nt <= 1) { for (size_t i = 0; i < n; i++) fn(i); return; }
  std::atomic<size_t> next{0};
  std::vector<std::thread> threads;
  for (unsigned t = 0; t < nt; t++) threads.emplace_back([&]() { for (size_t i; (i = next.fetch_add(1)) < n;) fn(i); });
  for (auto& t : threads) t.join();
}

// ---- ArgParser ----
namespace {
float parse_float(const std::string& s, const std::string& name) {
  char* end = nullptr; float v = strtof(s.c_str(), &end);
  if (end == s.c_str() || *end) fail("invalid value '%s' for '--%s': expected a number", s.c_str(), name.c_str());
  return v;
}
int parse_int(const std::string& s, const std::string& name) {
  char* end = nullptr; long v = strtol(s.c_str(), &end, 10);
  if (end == s.c_str() || *end) fail("invalid value '%s' for '--%s': expected an integer", s.c_str(), name.c_str());
  return (int)v;
}
}  // namespace

ArgParser& ArgParser::s(const char* name, const char* value, const char* help, std::string& out) {
  opts.push_back({name, value, help, 1, false, [&out](const std::vector<std::string>& v) { out = v[0]; }}); return *this;
}
ArgParser& ArgParser::f(const char* name, const char* value, const char* help, float& out) {
  std::string n = name;
  opts.push_back({name, value, help, 1, false, [&out, n](const std::vector<std::string>& v) { out = parse_float(v[0], n); }}); return *this;
}
ArgParser& ArgParser::i(const char* name, const char* value, const char* help, int& out) {
  std::string n = name;
  opts.push_back({name, value, help, 1, false, [&out, n](const std::vector<std::string>& v) { out = parse_int(v[0], n); }}); return *this;
}
ArgParser& ArgParser::b(const char* name, const char* help, bool& out) {
  opts.push_back({name, "", help, 0, false, [&out](const std::vector<std::string>&) { out = true; }}); return *this;
}
ArgParser& ArgParser::f3(const char* name, const char* value, const char* help, float* out) {
  std::string n = name;
  opts.push_back({name, value, help, 1, false, [out, n](const std::vector<std::string>& v) {
    if (sscanf(v[0].c_str(), "%f,%f,%f", out, out + 1, out + 2) != 3) fail("invalid value '%s' for '--%s': expected x,y,z", v[0].c_str(), n.c_str()); }});
  return *this;
}
ArgParser& ArgParser::multi(const char* name, const char* value, int nvals, const char* help, std::vector<std::vector<std::string>>& out) {
  opts.push_back({name, value, help, nvals, true, [&out](const std::vector<std::string>& v) { out.push_back(v); }}); return *this;
}
bool ArgParser::given(const char* name) const { for (auto& o : opts) if (o.name == name) return o.seen > 0; return false; }

std::string ArgParser::help() const {
  std::string s = usage + "\n\nOptions:\n";
  for (auto& o : opts) {
    std::string head = "      --" + o.name; if (!o.value_name.empty()) head += " " + o.value_name;
    s += head + "\n          " + o.help + "\n";
  }
  s += "      --help\n          Print help\n";
  return s;
}

bool ArgParser::parse(int argc, char** argv, int start) {
  for (int i = start; i < argc; i++) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") { fputs(help().c_str(), stdout); return false; }
    if (a.rfind("--", 0) != 0) fail("unexpected argument '%s' found", a.c_str());
    std::string name = a.substr(2), inline_value; bool has_inline = false;
    auto eq = name.find('='); if (eq != std::string::npos) { inline_value = name.substr(eq + 1); name = name.substr(0, eq); has_inline = true; }
    Opt* o = nullptr; for (auto& c : opts) if (c.name == name) { o = &c; break; }
    if (!o) fail("unexpected argument '--%s' found\n\nFor more information, try '--help'.", name.c_str());
    std::vector<std::string> vals;
    if (o->nvals == 0) { if (has_inline) fail("unexpected value for flag '--%s'", o->name.c_str()); }
    else if (has_inline) { if (o->nvals != 1) fail("'--%s' takes %d values", o->name.c_str(), o->nvals); vals.push_back(inline_value); }
    else {
      for (int k = 0; k < o->nvals; k++) { if (i + 1 >= argc) fail("a value is required for '--%s %s' but none was supplied", o->name.c_str(), o->value_name.c_str()); vals.push_back(argv[++i]); }
    }
    if (o->seen && !o->repeat) fail("the argument '--%s' cannot be used multiple times", o->name.c_str());
    o->seen++;
    o->set(vals);
  }
  return true;
}

}  // namespace b2c
