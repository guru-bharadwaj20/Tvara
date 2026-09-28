#pragma once
#include "tvara/mapped_file.hpp"
#include <cmath>

namespace tvara::gguf {
class TensorView {
public:
    TensorView(std::shared_ptr<const MappedFile> file,TensorInfo info,std::size_t data_offset)
        : file_(std::move(file)),info_(std::move(info)),data_offset_(data_offset) {
        if(!file_) throw Error("tensor view requires a mapping owner");
        const auto count=tensor_bytes(info_);
        const auto size=file_->bytes().size();
        if(data_offset_>size || info_.offset>size-data_offset_ || count>size-data_offset_-info_.offset)
            throw Error("tensor view exceeds its mapping");
    }
    const TensorInfo& info() const { return info_; }
    std::span<const std::byte> bytes() const {
        return file_->bytes().subspan(data_offset_+static_cast<std::size_t>(info_.offset),tensor_bytes(info_));
    }
private:
    std::shared_ptr<const MappedFile> file_;
    TensorInfo info_;
    std::size_t data_offset_;
};
class ModelFile {
public:
    explicit ModelFile(const std::filesystem::path& path)
        : file_(std::make_shared<MappedFile>(path)),document_(read_document(file_->bytes())) {}
    const Document& document() const { return document_; }
    TensorView tensor(const std::string& name) const {
        auto found=document_.tensors.find(name);
        if(found==document_.tensors.end()) throw Error("missing GGUF tensor: "+name);
        return {file_,found->second,document_.data_offset};
    }
private:
    std::shared_ptr<const MappedFile> file_;
    Document document_;
};
struct LlamaShape {
    std::uint64_t vocab=0,hidden=0,intermediate=0,layers=0,heads=0,kv_heads=0,context=0;
    double rms_epsilon=1e-6,rope_theta=10000;
};
inline void validate_llama_tensors(const Document& doc,const LlamaShape& shape) {
    if(!shape.vocab || !shape.hidden || !shape.intermediate || !shape.layers || !shape.heads ||
       !shape.kv_heads || shape.hidden%shape.heads || shape.heads%shape.kv_heads)
        throw Error("invalid Llama dimensions");
    const auto kv=shape.hidden/shape.heads*shape.kv_heads;
    auto require=[&](const std::string& name,std::vector<std::uint64_t> dimensions) {
        auto it=doc.tensors.find(name);
        if(it==doc.tensors.end() || it->second.dimensions!=dimensions)
            throw Error("missing or malformed Llama tensor: "+name);
    };
    require("token_embd.weight",{shape.hidden,shape.vocab});
    require("output_norm.weight",{shape.hidden});
    if(doc.tensors.contains("output.weight")) require("output.weight",{shape.hidden,shape.vocab});
    for(std::uint64_t i=0;i<shape.layers;++i) {
        const auto p="blk."+std::to_string(i)+".";
        require(p+"attn_norm.weight",{shape.hidden}); require(p+"ffn_norm.weight",{shape.hidden});
        require(p+"attn_q.weight",{shape.hidden,shape.hidden});
        require(p+"attn_k.weight",{shape.hidden,kv}); require(p+"attn_v.weight",{shape.hidden,kv});
        require(p+"attn_output.weight",{shape.hidden,shape.hidden});
        require(p+"ffn_gate.weight",{shape.hidden,shape.intermediate});
        require(p+"ffn_up.weight",{shape.hidden,shape.intermediate});
        require(p+"ffn_down.weight",{shape.intermediate,shape.hidden});
    }
}

inline LlamaShape llama_shape(const Document& doc) {
    auto text=[&](const std::string& name) -> const std::string& {
        auto it=doc.metadata.find(name);
        if(it==doc.metadata.end()) throw Error("missing metadata: "+name);
        auto value=std::get_if<std::string>(&it->second.data);
        if(!value) throw Error("metadata must be a string: "+name);
        return *value;
    };
    auto number=[&](const std::string& name) -> std::uint64_t {
        auto it=doc.metadata.find(name);
        if(it==doc.metadata.end()) throw Error("missing metadata: "+name);
        auto value=std::get_if<std::uint64_t>(&it->second.data);
        if(!value || !*value || *value>1000000) throw Error("invalid model dimension: "+name);
        return *value;
    };
    if(text("general.architecture")!="llama") throw Error("unsupported model architecture");
    auto vocab_it=doc.metadata.find("tokenizer.ggml.tokens");
    if(vocab_it==doc.metadata.end()) throw Error("missing tokenizer vocabulary");
    auto vocab=std::get_if<Value::Array>(&vocab_it->second.data);
    if(!vocab || vocab->empty()) throw Error("invalid tokenizer vocabulary");
    LlamaShape s{vocab->size(),number("llama.embedding_length"),number("llama.feed_forward_length"),
                 number("llama.block_count"),number("llama.attention.head_count"),
                 number("llama.attention.head_count_kv"),number("llama.context_length")};
    auto floating=[&](const std::string& name,double fallback) {
        auto it=doc.metadata.find(name);
        if(it==doc.metadata.end()) return fallback;
        auto value=std::get_if<double>(&it->second.data);
        if(!value || !std::isfinite(*value) || *value<=0) throw Error("invalid floating metadata: "+name);
        return *value;
    };
    s.rms_epsilon=floating("llama.attention.layer_norm_rms_epsilon",1e-6);
    s.rope_theta=floating("llama.rope.freq_base",10000);
    if(s.hidden%s.heads || s.heads%s.kv_heads || (s.hidden/s.heads)%2)
        throw Error("invalid Llama head layout");
    if(auto it=doc.metadata.find("llama.rope.scaling.type");it!=doc.metadata.end()) {
        auto value=std::get_if<std::string>(&it->second.data);
        if(!value || *value!="none") throw Error("scaled RoPE requires a separate implementation");
    }
    return s;
}

// MODEL_IMPLEMENTATIONS
}  // namespace tvara::gguf
