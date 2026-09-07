#include "util/log.h"
#include <cstdlib>
#include <vector>

namespace b2c {
static std::string vformat(const char* fmt, va_list ap) {
  va_list ap2; va_copy(ap2, ap);
  int n = vsnprintf(nullptr, 0, fmt, ap2); va_end(ap2);
  std::string s; s.resize(n > 0 ? n : 0);
  if (n > 0) vsnprintf(s.data(), n + 1, fmt, ap);
  return s;
}
std::string format(const char* fmt, ...) { va_list ap; va_start(ap, fmt); auto s = vformat(fmt, ap); va_end(ap); return s; }
void log_info(const char* fmt, ...) { va_list ap; va_start(ap, fmt); auto s = vformat(fmt, ap); va_end(ap); fprintf(stdout, "%s\n", s.c_str()); fflush(stdout); }
void log_warn(const char* fmt, ...) { va_list ap; va_start(ap, fmt); auto s = vformat(fmt, ap); va_end(ap); fprintf(stdout, "⚠️: %s\n", s.c_str()); fflush(stdout); }
void fail(const char* fmt, ...) { va_list ap; va_start(ap, fmt); auto s = vformat(fmt, ap); va_end(ap); throw std::runtime_error(s); }
double now_seconds() { using namespace std::chrono; return duration<double>(steady_clock::now().time_since_epoch()).count(); }
std::string format_duration(double secs) {
  if (secs < 0) secs = 0;
  int s = (int)(secs + 0.5);
  if (s < 60) return format("%ds", s);
  if (s < 3600) return format("%dm %ds", s / 60, s % 60);
  return format("%dh %dm", s / 3600, (s % 3600) / 60);
}
std::string format_count(size_t n) {
  if (n < 10000) return format("%zu", n);
  if (n < 1000000) return format("%.1fk", n / 1000.0);
  return format("%.2fM", n / 1e6);
}
}  // namespace b2c
