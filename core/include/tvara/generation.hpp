#pragma once
#include "tvara/model.hpp"
#include "tvara/sampling.hpp"
#include <optional>
namespace tvara {
struct GenerationOptions {
    std::size_t max_new_tokens=16,top_k=0;
    double temperature=0,top_p=1,min_p=0;
    float repetition=1,frequency=0,presence=0;
    std::uint64_t seed=0;
    std::optional<std::size_t> eos;
};
inline std::vector<std::size_t> generate(const ModelExecutor& model,std::span<const std::size_t> prompt,const GenerationOptions& options) {
    if(prompt.empty() || prompt.size()>model.config().context || !std::isfinite(options.temperature) || options.temperature<0 ||
       !std::isfinite(options.top_p) || options.top_p<=0 || options.top_p>1 ||
       !std::isfinite(options.min_p) || options.min_p<0 || options.min_p>1 ||
       (options.eos && *options.eos>=model.config().vocab))throw std::invalid_argument("invalid generation options");
    auto state=model.new_sequence();auto logits=model.prefill(prompt,state);
    std::mt19937_64 rng(options.seed);std::vector<std::size_t> history(prompt.begin(),prompt.end()),out;
    for(std::size_t i=0;i<options.max_new_tokens;++i) {
        if(state.length>=model.config().context)break;
        apply_penalties(logits,history,options.repetition,options.frequency,options.presence);
        std::size_t token;
        if(options.temperature==0)token=greedy(logits);
        else {auto p=probabilities(logits,options.temperature);filter_probabilities(p,options.top_k,options.top_p,options.min_p);
              std::discrete_distribution<std::size_t> distribution(p.begin(),p.end());token=distribution(rng);}
        out.push_back(token);history.push_back(token);
        if(options.eos && token==*options.eos)break;
        if(i+1<options.max_new_tokens)logits=model.decode(token,state);
    }
    return out;
}
}  // namespace tvara
