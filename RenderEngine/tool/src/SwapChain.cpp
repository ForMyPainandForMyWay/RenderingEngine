//
// Created by 冬榆 on 2026/2/22.
//

#include "SwapChain.hpp"
#include "Film.hpp"

SwapChain::SwapChain(const size_t width, const size_t height){
    for (int i = 0; i < 3; i++) {
        auto frame = std::make_shared<Film>(width, height);
        emptyQueue.push_back(frame); // 初始时所有 buffer 都是空的
    }
    isRunning = true;
}

// 申请渲染缓冲区
std::shared_ptr<Film> SwapChain::acquireBackBuffer() {
    std::unique_lock lock(mutex);
    Producer.wait(lock, [this] { return !emptyQueue.empty() || !isRunning; });
    if (!isRunning) return nullptr;
    auto frame = emptyQueue.front();
    emptyQueue.pop_front();
    return frame;
}

// 提交渲染完成的帧缓冲
void SwapChain::commitBackBuffer(const std::shared_ptr<Film> &frame) {
    {
        std::lock_guard lock(mutex);
        if (!pendingQueue.empty()) {
            emptyQueue.push_back(pendingQueue.front()); // 回收旧的
            pendingQueue.pop_front(); // 清除旧的引用
        }
        pendingQueue.push_back(frame);
    }
    Consumer.notify_one();
}

// 申请渲染好的帧，如果没有，就返回空
std::shared_ptr<Film> SwapChain::acquireFrontBuffer() {
    std::unique_lock lock(mutex);
    // 阻塞等待有新帧或结束
    Consumer.wait(lock, [this] { return !pendingQueue.empty() || !isRunning; });
    if (pendingQueue.empty() && !isRunning) return nullptr;
    auto frame = pendingQueue.front();
    pendingQueue.pop_front();
    return frame;
}

// 归还缓冲区，将使用完毕的缓冲区复用
void SwapChain::releaseFrontBuffer(const std::shared_ptr<Film> &frame) {
    {
        std::lock_guard lock(mutex);
        emptyQueue.push_back(frame);
    }
    Producer.notify_one();
}

void SwapChain::stop() {
    {
        std::lock_guard lock(mutex);
        isRunning = false;
    }
    Producer.notify_all();
    Consumer.notify_all();
}