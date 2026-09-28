#pragma once
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace tvara::gguf {
class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
class Reader {
public:
    explicit Reader(std::span<const std::byte> data) : data_(data) {}
    std::size_t position() const { return position_; }
    std::size_t remaining() const { return data_.size() - position_; }
    std::span<const std::byte> take(std::size_t count) {
        if (count > remaining()) throw Error("truncated GGUF data");
        auto result = data_.subspan(position_, count);
        position_ += count;
        return result;
    }
    std::uint64_t uint_le(std::size_t width) {
        if (width == 0 || width > 8) throw Error("invalid integer width");
        auto bytes = take(width);
        std::uint64_t value = 0;
        for (std::size_t i = 0; i < width; ++i)
            value |= std::uint64_t(std::to_integer<unsigned char>(bytes[i])) << (8 * i);
        return value;
    }
    float f32() { return std::bit_cast<float>(static_cast<std::uint32_t>(uint_le(4))); }
    double f64() { return std::bit_cast<double>(uint_le(8)); }
private:
    std::span<const std::byte> data_;
    std::size_t position_ = 0;
};

struct Header { std::uint32_t version = 0; };
inline Header read_header(Reader& reader) {
    if (reader.uint_le(4) != 0x46554747) throw Error("invalid GGUF magic or unsupported byte order");
    auto version = static_cast<std::uint32_t>(reader.uint_le(4));
    if (version != 2 && version != 3) throw Error("unsupported GGUF version");
    return {version};
}

// IMPLEMENTATIONS
}  // namespace tvara::gguf
