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

struct Header {
    std::uint32_t version = 0;
    std::uint64_t tensor_count = 0;
    std::uint64_t metadata_count = 0;
};
inline Header read_header(Reader& reader) {
    if (reader.uint_le(4) != 0x46554747) throw Error("invalid GGUF magic or unsupported byte order");
    auto version = static_cast<std::uint32_t>(reader.uint_le(4));
    if (version != 2 && version != 3) throw Error("unsupported GGUF version");
    const auto tensors = reader.uint_le(8);
    const auto metadata = reader.uint_le(8);
    if (tensors > 1000000 || metadata > 1000000) throw Error("GGUF count limit exceeded");
    if (tensors > reader.remaining()/24 || metadata > reader.remaining()/13)
        throw Error("GGUF counts exceed available bytes");
    return {version, tensors, metadata};
}

struct Value {
    using Array = std::vector<Value>;
    std::variant<std::uint64_t, std::int64_t, double, bool, std::string, Array> data;
};
inline Value read_scalar(Reader& r, std::uint32_t type) {
    switch (type) {
        case 0: return {r.uint_le(1)};
        case 1: return {std::int64_t(std::bit_cast<std::int8_t>(static_cast<std::uint8_t>(r.uint_le(1))))};
        case 2: return {r.uint_le(2)};
        case 3: return {std::int64_t(std::bit_cast<std::int16_t>(static_cast<std::uint16_t>(r.uint_le(2))))};
        case 4: return {r.uint_le(4)};
        case 5: return {std::int64_t(std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(r.uint_le(4))))};
        case 6: return {double(r.f32())};
        case 7: {
            const auto b = r.uint_le(1);
            if (b > 1) throw Error("invalid boolean metadata");
            return {b != 0};
        }
        case 10: return {r.uint_le(8)};
        case 11: return {std::bit_cast<std::int64_t>(r.uint_le(8))};
        case 12: return {r.f64()};
        default: throw Error("unsupported scalar metadata type");
    }
}

inline std::string read_string(Reader& r) {
    const auto count = r.uint_le(8);
    if (count > 16*1024*1024 || count > r.remaining()) throw Error("GGUF string length exceeds limits");
    const auto bytes = r.take(static_cast<std::size_t>(count));
    return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}
inline Value read_value(Reader& r, std::uint32_t type) {
    if (type == 8) return {read_string(r)};
    if (type != 9) return read_scalar(r,type);
    const auto element_type = static_cast<std::uint32_t>(r.uint_le(4));
    const auto count = r.uint_le(8);
    if (element_type == 9 || element_type > 12 || count > 1000000 || count > r.remaining())
        throw Error("invalid or oversized metadata array");
    Value::Array values;
    values.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i=0; i<count; ++i) values.push_back(read_value(r,element_type));
    return {std::move(values)};
}
using Metadata = std::map<std::string,Value>;
inline Metadata read_metadata(Reader& r, std::uint64_t count) {
    Metadata values;
    for (std::uint64_t i=0; i<count; ++i) {
        auto name=read_string(r);
        auto value=read_value(r,static_cast<std::uint32_t>(r.uint_le(4)));
        if (name.empty() || !values.emplace(std::move(name),std::move(value)).second)
            throw Error("empty or duplicate GGUF metadata key");
    }
    return values;
}

// IMPLEMENTATIONS
}  // namespace tvara::gguf
