#pragma once
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace tvara {
inline std::size_t element_count(std::span<const std::size_t> shape) {
    if (shape.empty()) throw std::invalid_argument("tensor rank must be positive");
    std::size_t count = 1;
    for (auto dim : shape) {
        if (dim == 0) throw std::invalid_argument("tensor dimensions must be positive");
        if (count > std::numeric_limits<std::size_t>::max() / dim)
            throw std::overflow_error("tensor shape overflow");
        count *= dim;
    }
    return count;
}
class Tensor {
public:
    explicit Tensor(std::vector<std::size_t> shape)
        : shape_(std::move(shape)), data_(element_count(shape_)) {}
    std::span<const std::size_t> shape() const { return shape_; }
    std::span<float> data() { return data_; }
    std::span<const float> data() const { return data_; }
    float& at(std::size_t i) { return data_.at(i); }
    const float& at(std::size_t i) const { return data_.at(i); }
private:
    std::vector<std::size_t> shape_;
    std::vector<float> data_;
};
}  // namespace tvara
