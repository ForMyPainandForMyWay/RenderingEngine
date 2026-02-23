//
// Created by 冬榆 on 2026/2/22.
//

#include "SwapChain.hpp"
#include "Film.hpp"

SwapChain::SwapChain(const size_t width, const size_t height) : w(width), h(height) {
    for (int i = 0; i < 3; i++) {
        auto frame = std::make_shared<Film>(height, height);
        emptyQueue.push_back(frame); // 初始时所有 buffer 都是空的
    }
    isRunning = true;
}

std::shared_ptr<Film> SwapChain::acquireBackBuffer() {
    std::unique_lock lock(mutex);
    Producer.wait(lock, [this] { return !emptyQueue.empty() || !isRunning; });
    if (!isRunning) return nullptr;
    auto frame = emptyQueue.back();
    emptyQueue.pop_back();
    return frame;
}

void SwapChain::commitBackBuffer(const std::shared_ptr<Film> &frame) {
    {
        std::lock_guard lock(mutex);
        if (!pendingQueue.empty()) {
            emptyQueue.push_back(pendingQueue.front()); // 回收旧的
            pendingQueue.clear(); // 清除旧的引用
        }
        pendingQueue.push_back(frame);
    }
    Consumer.notify_one();
}

// 返回渲染好的帧，如果没有，就返回空
std::shared_ptr<Film> SwapChain::acquireFrontBuffer() {
    std::unique_lock lock(mutex);
    // 阻塞等待有新帧或结束
    Consumer.wait(lock, [this] { return !pendingQueue.empty() || !isRunning; });
    if (pendingQueue.empty() && !isRunning) return nullptr;
    auto frame = pendingQueue.back();
    pendingQueue.pop_back();
    return frame;
}

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