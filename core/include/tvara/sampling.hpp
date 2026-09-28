#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>
namespace tvara {
inline std::size_t greedy(std::span<const float> logits) {
    if(logits.empty())throw std::invalid_argument("empty logits");
    bool valid=false;
    for(auto value:logits) {if(std::isnan(value) || value==std::numeric_limits<float>::infinity())throw std::invalid_argument("invalid logits");valid|=std::isfinite(value);}
    if(!valid)throw std::invalid_argument("all logits masked");
    return static_cast<std::size_t>(std::max_element(logits.begin(),logits.end())-logits.begin());
}
inline std::vector<double> probabilities(std::span<const float> logits,double temperature) {
    if(!std::isfinite(temperature) || temperature<=0)throw std::invalid_argument("temperature must be positive");
    const auto best=greedy(logits);const double maximum=logits[best];
    std::vector<double> p(logits.size());double sum=0;
    for(std::size_t i=0;i<p.size();++i){p[i]=std::exp((double(logits[i])-maximum)/temperature);sum+=p[i];}
    for(auto& value:p)value/=sum;
    return p;
}
inline std::size_t sample(std::span<const float> logits,double temperature,std::mt19937_64& rng) {
    if(temperature==0)return greedy(logits);
    auto p=probabilities(logits,temperature);std::discrete_distribution<std::size_t> distribution(p.begin(),p.end());
    return distribution(rng);
}
// SAMPLE_CODE
}  // namespace tvara
