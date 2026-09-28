#include "tvara/gguf.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <functional>
#include <cmath>

using namespace tvara::gguf;
using Bytes = std::vector<std::byte>;
void check(bool condition) { if (!condition) throw std::runtime_error("GGUF check failed"); }
void rejects(const std::function<void()>& fn) {
    bool rejected = false;
    try { fn(); } catch (const Error&) { rejected = true; }
    check(rejected);
}
void integer(Bytes& out, std::uint64_t value, std::size_t width) {
    for (std::size_t i=0; i<width; ++i) out.push_back(std::byte((value >> (8*i)) & 255));
}
void string_bytes(Bytes& out, const std::string& value) {
    integer(out, value.size(), 8);
    for (unsigned char c : value) out.push_back(std::byte(c));
}
void test_reader() {
    Bytes data; integer(data, 0x01020304, 4); integer(data, std::bit_cast<std::uint32_t>(1.5f), 4);
    Reader reader(data);
    check(reader.uint_le(4) == 0x01020304);
    check(reader.f32() == 1.5f);
    check(reader.remaining() == 0);
    rejects([&]{ reader.take(1); });
    rejects([&]{ reader.uint_le(9); });
}

// EXTRA_TESTS
int main() {
    try {
        test_reader();
        // RUN_TESTS
        std::cout << "GGUF validation checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
