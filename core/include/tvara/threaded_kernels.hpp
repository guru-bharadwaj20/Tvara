#pragma once
#include "tvara/thread_pool.hpp"
#include "tvara/quant.hpp"
namespace tvara {
inline void threaded_matvec(ThreadPool& pool,std::span<const float> weights,std::size_t rows,std::size_t cols,
                            std::span<const float> x,std::span<float> out){
    if(!rows || !cols || rows>std::numeric_limits<std::size_t>::max()/cols || weights.size()!=rows*cols || x.size()!=cols || out.size()!=rows)
        throw std::invalid_argument("invalid threaded matrix dimensions");
    pool.execute([&](std::size_t worker){auto [start,end]=partition_rows(rows,worker,pool.size());
        for(auto r=start;r<end;++r)out[r]=quant::dot_f32(weights.subspan(r*cols,cols),x);});
}
}
