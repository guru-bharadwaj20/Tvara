#pragma once
#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
namespace tvara {
class ThreadPool {
public:
    explicit ThreadPool(std::size_t count):count_(count){
        if(!count || count>256)throw std::invalid_argument("worker count must be 1..256");
        try {for(std::size_t i=0;i<count;++i)workers_.emplace_back([this,i]{worker(i);});}
        catch(...){shutdown();throw;}
    }
    ~ThreadPool(){shutdown();}
    ThreadPool(const ThreadPool&)=delete;
    ThreadPool& operator=(const ThreadPool&)=delete;
    std::size_t size()const{return count_;}
    void execute(std::function<void(std::size_t)> job){
        if(current_pool_==this)throw std::logic_error("nested pool execution is unsupported");
        auto callback=std::make_shared<const std::function<void(std::size_t)>>(std::move(job));
        std::lock_guard serial(run_mutex_);std::unique_lock lock(mutex_);
        if(stopping_)throw std::logic_error("pool is stopping");
        callback_=std::move(callback);remaining_=count_;error_=nullptr;++generation_;
        ready_.notify_all();finished_.wait(lock,[&]{return remaining_==0;});callback_.reset();
        if(error_)std::rethrow_exception(error_);
    }
private:
    void shutdown()noexcept{
        {std::lock_guard lock(mutex_);stopping_=true;}ready_.notify_all();
        for(auto& worker:workers_)if(worker.joinable())worker.join();
    }
    void worker(std::size_t index){
        current_pool_=this;std::size_t seen=0;
        while(true){std::unique_lock lock(mutex_);ready_.wait(lock,[&]{return stopping_ || generation_!=seen;});
            if(stopping_)return;
            seen=generation_;auto callback=callback_;lock.unlock();std::exception_ptr error;
            try{(*callback)(index);}catch(...){error=std::current_exception();}
            lock.lock();if(error && !error_)error_=error;if(--remaining_==0)finished_.notify_one();
        }
    }
    inline static thread_local const ThreadPool* current_pool_=nullptr;
    std::size_t count_,remaining_=0,generation_=0;
    bool stopping_=false;
    std::vector<std::thread> workers_;
    std::mutex mutex_,run_mutex_;
    std::condition_variable ready_,finished_;
    std::shared_ptr<const std::function<void(std::size_t)>> callback_;
    std::exception_ptr error_;
};
inline std::pair<std::size_t,std::size_t> partition_rows(std::size_t rows,std::size_t index,std::size_t workers) {
    if(!workers || index>=workers)throw std::invalid_argument("invalid row partition");
    const auto base=rows/workers,remainder=rows%workers;
    const auto start=base*index+std::min(index,remainder);return {start,start+base+(index<remainder?1:0)};
}

class SpinBarrier {
public:
    explicit SpinBarrier(std::size_t count):count_(count),remaining_(count){if(!count)throw std::invalid_argument("empty barrier");}
    void arrive_and_wait(){
        const auto seen=generation_.load(std::memory_order_acquire);
        if(remaining_.fetch_sub(1,std::memory_order_acq_rel)==1){
            {std::lock_guard lock(mutex_);remaining_.store(count_,std::memory_order_relaxed);generation_.fetch_add(1,std::memory_order_release);}
            changed_.notify_all();return;
        }
        for(int spin=0;spin<64;++spin){if(generation_.load(std::memory_order_acquire)!=seen)return;std::this_thread::yield();}
        std::unique_lock lock(mutex_);changed_.wait(lock,[&]{return generation_.load(std::memory_order_acquire)!=seen;});
    }
private:
    const std::size_t count_;
    std::atomic<std::size_t> remaining_,generation_{0};
    std::mutex mutex_;std::condition_variable changed_;
};

// THREAD_CODE
}
