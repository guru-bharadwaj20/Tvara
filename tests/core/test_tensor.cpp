#include "tvara/tensor.hpp"
#include "tvara/version.hpp"
#include <iostream>
#include <string>

int main() {
    tvara::Tensor tensor({2, 3});
    if (tensor.data().size() != 6 || tensor.at(5) != 0.0f) return 1;
    tensor.at(5) = 9.0f;
    if (tensor.data()[5] != 9.0f) return 2;
    bool zero = false, overflow = false, bounds = false;
    try { tvara::Tensor invalid({2, 0}); } catch (const std::invalid_argument&) { zero = true; }
    try { tvara::Tensor invalid({std::numeric_limits<std::size_t>::max(), 2}); }
    catch (const std::overflow_error&) { overflow = true; }
    try { tensor.at(6); } catch (const std::out_of_range&) { bounds = true; }
    if (!(zero && overflow && bounds)) return 3;
    if (std::string(tvara::version()).empty()) return 4;
    std::cout << "Tensor shape, overflow, bounds, and ownership checks passed\n";
}
