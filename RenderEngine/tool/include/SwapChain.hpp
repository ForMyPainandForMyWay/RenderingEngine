//
// Created by 冬榆 on 2026/2/22.
//

#ifndef RENDERINGENGINE_SWAPCHAIN_HPP
#define RENDERINGENGINE_SWAPCHAIN_HPP

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>


struct Film;
struct Pixel;

class SwapChain {
public:
    SwapChain(size_t width, size_t height);
    std::shared_ptr<Film> acquireBackBuffer();
    void commitBackBuffer(const std::shared_ptr<Film> &frame);
    std::shared_ptr<Film> acquireFrontBuffer();
    void releaseFrontBuffer(const std::shared_ptr<Film> &frame);
    void stop();

private:
    std::atomic<bool> isRunning;
    std::mutex mutex;
    std::condition_variable Producer;
    std::condition_variable Consumer;

    std::deque<std::shared_ptr<Film>> emptyQueue;
    std::deque<std::shared_ptr<Film>> pendingQueue;
};


#endif //RENDERINGENGINE_SWAPCHAIN_HPP