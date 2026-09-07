#pragma once
#include <cstdio>
#include <cstdarg>
#include <string>
#include <stdexcept>
#include <chrono>

namespace b2c {
void log_info(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void log_warn(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
std::string format(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
[[noreturn]] void fail(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
double now_seconds();
std::string format_duration(double secs);
std::string format_count(size_t n);
}  // namespace b2c
