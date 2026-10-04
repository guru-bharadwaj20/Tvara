#pragma once
#include "tvara/model.hpp"
#include <chrono>
#include <iostream>
#include <functional>
inline void require(bool condition) { if(!condition) throw std::runtime_error("prefill check failed"); }
inline void close_vectors(std::span<const float> a,std::span<const float> b,float tolerance=1e-5f) {
    require(a.size()==b.size());for(std::size_t i=0;i<a.size();++i)require(std::abs(a[i]-b[i])<=tolerance*(1+std::abs(b[i])));
}
inline std::vector<float> signal(std::size_t n,float scale=1) {
    std::vector<float> out(n);for(std::size_t i=0;i<n;++i)out[i]=scale*std::sin(static_cast<float>(i)*0.173f);return out;
}
inline double median_us(const std::function<void()>& call,int repeats=9) {
    std::vector<double> times;call();for(int i=0;i<repeats;++i){auto begin=std::chrono::steady_clock::now();call();times.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count());}
    std::sort(times.begin(),times.end());return times[times.size()/2];
}
inline float max_error(std::span<const float> a,std::span<const float> b){require(a.size()==b.size());float error=0;for(std::size_t i=0;i<a.size();++i)error=std::max(error,std::abs(a[i]-b[i]));return error;}
inline void rejects(const std::function<void()>& f){bool bad=false;try{f();}catch(const std::invalid_argument&){bad=true;}require(bad);}
