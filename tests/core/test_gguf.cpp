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

void test_header() {
    for (std::uint32_t version : {2u, 3u}) {
        Bytes data; integer(data,0x46554747,4); integer(data,version,4); integer(data,0,8); integer(data,0,8);
        Reader reader(data); check(read_header(reader).version == version);
    }
    Bytes invalid; integer(invalid,0x46554747,4); integer(invalid,1,4);
    Reader reader(invalid); rejects([&]{ read_header(reader); });
    Bytes wrong(8); Reader bad(wrong); rejects([&]{ read_header(bad); });
}

void test_counts() {
    Bytes data; integer(data,0x46554747,4); integer(data,3,4);
    integer(data,1000001,8); integer(data,0,8);
    Reader reader(data); rejects([&]{ read_header(reader); });
}
void test_scalars() {
    Bytes data; integer(data,255,1); integer(data,std::bit_cast<std::uint32_t>(2.5f),4); integer(data,1,1);
    Reader r(data);
    check(std::get<std::int64_t>(read_scalar(r,1).data)==-1);
    check(std::get<double>(read_scalar(r,6).data)==2.5);
    check(std::get<bool>(read_scalar(r,7).data));
    Bytes invalid{std::byte(2)}; Reader bad(invalid); rejects([&]{ read_scalar(bad,7); });
}

void test_metadata() {
    Bytes data; string_bytes(data,"general.name"); integer(data,8,4); string_bytes(data,"fixture");
    string_bytes(data,"numbers"); integer(data,9,4); integer(data,4,4); integer(data,2,8);
    integer(data,5,4); integer(data,7,4);
    Reader r(data); auto values=read_metadata(r,2);
    check(std::get<std::string>(values.at("general.name").data)=="fixture");
    const auto& array=std::get<Value::Array>(values.at("numbers").data);
    check(array.size()==2 && std::get<std::uint64_t>(array[1].data)==7);
    Bytes bad; integer(bad,100,8); Reader truncated(bad); rejects([&]{ read_string(truncated); });
}

void test_tensor_dimensions() {
    Bytes data; string_bytes(data,"weight"); integer(data,2,4); integer(data,32,8); integer(data,4,8); integer(data,8,4); integer(data,64,8);
    Reader r(data); auto t=read_tensor_info(r);
    check(t.type==8 && t.offset==64);
    check(t.name=="weight" && t.dimensions==std::vector<std::uint64_t>({32,4}));
    Bytes bad; string_bytes(bad,"weight"); integer(bad,5,4);
    Reader invalid(bad); rejects([&]{ read_tensor_info(invalid); });
}

Bytes one_tensor_file(std::uint64_t offset=0) {
    Bytes out; integer(out,0x46554747,4); integer(out,3,4); integer(out,1,8); integer(out,0,8);
    string_bytes(out,"weight"); integer(out,1,4); integer(out,4,8); integer(out,0,4); integer(out,offset,8);
    out.resize(align_up(out.size(),32)+16);
    return out;
}
void test_alignment() {
    auto bytes=one_tensor_file(); auto doc=read_document(bytes);
    check(doc.data_offset%32==0);
    rejects([]{ align_up(5,3); });
    auto bad=one_tensor_file(1); rejects([&]{ read_document(bad); });
}

void test_payload_bounds() {
    auto bytes=one_tensor_file(1024); rejects([&]{ read_document(bytes); });
    auto truncated=one_tensor_file(); truncated.pop_back(); rejects([&]{ read_document(truncated); });
    TensorInfo invalid{"bad",{31},8,0}; rejects([&]{ tensor_bytes(invalid); });
    TensorInfo overflow{"bad",{std::numeric_limits<std::uint64_t>::max(),2},0,0};
    rejects([&]{ tensor_bytes(overflow); });
}

// EXTRA_TESTS
int main() {
    try {
        test_reader();
        test_header();
        test_counts();
        test_scalars();
        test_metadata();
        test_tensor_dimensions();
        test_alignment();
        test_payload_bounds();
        // RUN_TESTS
        std::cout << "GGUF validation checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
