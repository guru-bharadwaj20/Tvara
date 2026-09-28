#pragma once
#include "tvara/mapped_file.hpp"

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
// MODEL_IMPLEMENTATIONS
}  // namespace tvara::gguf
