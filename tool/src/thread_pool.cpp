//
// Created by yyd on 2025/12/23.
//

#include "thread_pool.hpp"

//子线程调度
void ThreadPool::WorkerThread(ThreadPool *master) {
    while (true) {
        // 尝试获取任务
        if (std::function<void()> task; master->popTask(task)) {
            task();  // 执行任务
            --master->pending_task_count;
        } else return;
    }
}

// 主动等待所有任务完成
void ThreadPool::wait() const {
    while (this->pending_task_count > 0) {
        std::this_thread::yield();
    }
}

ThreadPool::ThreadPool(size_t thread_count) {
    if (thread_count == 0) {
        thread_count = std::thread::hardware_concurrency();
    }
    for (size_t i = 1; i <= thread_count; i++) {
        threads.emplace_back(WorkerThread, this);
    }
}

ThreadPool::~ThreadPool() {
    wait();
    {
        std::lock_guard guard(lock);
        alive = false;
    }
    cv.notify_all();  // 唤醒所有线程避免忙等
    for (auto &thread : threads) {
        if (thread.joinable()) thread.join();
    }
    threads.clear();
}

bool ThreadPool::popTask(std::function<void()>& task) {
    std::unique_lock u_lock(lock);
    cv.wait(u_lock, [this] {
        return !tasks.empty() || !alive;
    });
    if (tasks.empty()) return false;
    // 移动语义：将任务从队列中 move 出来，避免拷贝 lambda
    task = std::move(tasks.front());
    tasks.pop_front();
    return true;
}