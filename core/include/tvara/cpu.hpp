#pragma once
namespace tvara {
struct CpuFeatures {bool avx2=false,avx512=false;};
inline CpuFeatures cpu_features() {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return {bool(__builtin_cpu_supports("avx2")),bool(__builtin_cpu_supports("avx512f"))};
#else
    return {};
#endif
}
}
