#include "prefill_support.hpp"
#include "tvara/prefill.hpp"
int main(){const std::size_t rows=19,cols=64,tokens=11;std::vector<std::byte> w(rows*(cols/32)*34);
    for(std::size_t b=0;b<w.size()/34;++b){w[b*34]=std::byte(0);w[b*34+1]=std::byte(0x3c);for(std::size_t k=0;k<32;++k)w[b*34+2+k]=std::byte((b+k)%21);}
    auto x=signal(tokens*cols);auto tile=tvara::quantize_prefill(x,tokens,cols);auto output=tvara::prefill_q8(w,rows,tile);auto scalar=tvara::prefill_q8(w,rows,tile,true);close_vectors(output,scalar);
    tvara::Matrix decoded(rows,cols);tvara::quant::dequantize_q8_0(w,decoded.values);auto reconstructed=x;
    for(std::size_t b=0;b<tile.blocks.size();++b)for(std::size_t k=0;k<32;++k)reconstructed[b*32+k]=tile.blocks[b].scale*tile.blocks[b].values[k];
    auto reference=tvara::prefill_reference(decoded,reconstructed,tokens);close_vectors(output,reference,2e-5f);
    auto reused=median_us([&]{output=tvara::prefill_q8(w,rows,tile);});auto quantization=median_us([&]{tile=tvara::quantize_prefill(x,tokens,cols);});
    std::cout<<"{\"quantized_blocks_once\":"<<tile.blocks.size()<<",\"max_error\":"<<max_error(output,reference)<<",\"reused_us\":"<<reused<<",\"quantization_us\":"<<quantization<<",\"scope\":\"Q8 kernel API; model loader remains F32\"}";
}
