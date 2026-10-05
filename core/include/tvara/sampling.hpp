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
inline void filter_probabilities(std::vector<double>& p,std::size_t top_k=0,double top_p=1,double min_p=0) {
    if(p.empty() || !std::isfinite(top_p) || top_p<=0 || top_p>1 || !std::isfinite(min_p) || min_p<0 || min_p>1)
        throw std::invalid_argument("invalid sampler filters");
    double maximum=0;for(auto v:p){if(!std::isfinite(v)||v<0)throw std::invalid_argument("invalid probability");maximum=std::max(maximum,v);}
    if(maximum<=0)throw std::invalid_argument("empty distribution");
    double total=0;for(auto& v:p){v/=maximum;total+=v;}
    for(auto& v:p)v/=total;
    std::vector<std::size_t> order(p.size());std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return p[a]>p[b];});
    if(top_k && top_k<order.size())for(std::size_t i=top_k;i<order.size();++i)p[order[i]]=0;
    const double cutoff=p[order[0]]*min_p;
    for(auto& v:p)if(v<cutoff)v=0;
    total=std::accumulate(p.begin(),p.end(),0.0);for(auto& v:p)v/=total;
    double cumulative=0;
    for(auto index:order){if(cumulative>=top_p)p[index]=0;else cumulative+=p[index];}
    total=std::accumulate(p.begin(),p.end(),0.0);for(auto& v:p)v/=total;
}

inline void apply_penalties(std::span<float> logits,std::span<const std::size_t> history,
                            float repetition=1,float frequency=0,float presence=0) {
    if(!std::isfinite(repetition) || repetition<=0 || !std::isfinite(frequency) || !std::isfinite(presence))
        throw std::invalid_argument("invalid penalties");
    std::vector<std::size_t> counts(logits.size());
    for(auto token:history){if(token>=logits.size())throw std::invalid_argument("history token outside vocabulary");++counts[token];}
    for(std::size_t i=0;i<logits.size();++i)if(counts[i]) {
        logits[i]=logits[i]>=0?logits[i]/repetition:logits[i]*repetition;
        logits[i]-=frequency*static_cast<float>(counts[i])+presence;
    }
}

// SAMPLE_CODE
}  // namespace tvara
