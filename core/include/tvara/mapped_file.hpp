#pragma once
#include "tvara/gguf.hpp"
#include <filesystem>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace tvara::gguf {
class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path) {
#ifdef _WIN32
        file_=CreateFileW(path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
        if(file_==INVALID_HANDLE_VALUE) throw Error("cannot open model file");
        LARGE_INTEGER size;
        if(!GetFileSizeEx(file_,&size) || size.QuadPart<=0 ||
           static_cast<unsigned long long>(size.QuadPart)>std::numeric_limits<std::size_t>::max()) {
            cleanup(); throw Error("invalid model file size");
        }
        size_=static_cast<std::size_t>(size.QuadPart);
        mapping_=CreateFileMappingW(file_,nullptr,PAGE_READONLY,0,0,nullptr);
        if(!mapping_) { cleanup(); throw Error("cannot map model file"); }
        data_=static_cast<const std::byte*>(MapViewOfFile(mapping_,FILE_MAP_READ,0,0,0));
        if(!data_) { cleanup(); throw Error("cannot view model mapping"); }
#else
        file_=open(path.c_str(),O_RDONLY);
        if(file_<0) throw Error("cannot open model file");
        struct stat info{};
        if(fstat(file_,&info)!=0 || info.st_size<=0 ||
           static_cast<std::uintmax_t>(info.st_size)>std::numeric_limits<std::size_t>::max()) {
            cleanup(); throw Error("invalid model file size");
        }
        size_=static_cast<std::size_t>(info.st_size);
        auto pointer=mmap(nullptr,size_,PROT_READ,MAP_PRIVATE,file_,0);
        if(pointer==MAP_FAILED) { cleanup(); throw Error("cannot map model file"); }
        data_=static_cast<const std::byte*>(pointer);
#endif
    }
    ~MappedFile() { cleanup(); }
    MappedFile(const MappedFile&)=delete;
    MappedFile& operator=(const MappedFile&)=delete;
    std::span<const std::byte> bytes() const { return {data_,size_}; }
private:
    void cleanup() noexcept {
#ifdef _WIN32
        if(data_) UnmapViewOfFile(data_);
        if(mapping_) CloseHandle(mapping_);
        if(file_!=INVALID_HANDLE_VALUE) CloseHandle(file_);
        mapping_=nullptr; file_=INVALID_HANDLE_VALUE;
#else
        if(data_) munmap(const_cast<std::byte*>(data_),size_);
        if(file_>=0) close(file_);
        file_=-1;
#endif
        data_=nullptr;
    }
    const std::byte* data_=nullptr;
    std::size_t size_=0;
#ifdef _WIN32
    HANDLE file_=INVALID_HANDLE_VALUE;
    HANDLE mapping_=nullptr;
#else
    int file_=-1;
#endif
};
}  // namespace tvara::gguf
