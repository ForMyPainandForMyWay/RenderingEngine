//
// Created by yyd on 2025/12/23.
//

#include "thread_pool.h"

//子线程调度
void ThreadPool::WorkerThread(ThreadPool *master) {
    while (master->alive) {

        if (Task *task = master->getTask(); task != nullptr) {

            task->run();
            delete task;
            --master->pending_task_count;
        } else {
            std::this_thread::yield();
        }
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
    alive = false;
    for (auto &thread : threads) {
        thread.join();
    }
    threads.clear();
}

void ThreadPool::addTask(Task *task) {
    std::lock_guard guard(lock);
    tasks.push_back(task);
    ++this->pending_task_count;
}

Task* ThreadPool::getTask() {
    std::lock_guard guard(lock);
    if (tasks.empty()) {
        return nullptr;
    }
    Task* task = tasks.front();
    tasks.pop_front();
    return task;
}