//
// Created by yyd on 2025/12/23.
//

#ifndef UNTITLED_THREAD_POOL_H
#define UNTITLED_THREAD_POOL_H
#include <list>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>

class Task {
public:
    virtual ~Task() = default;

    virtual void run() = 0;
};


class FuncTask : public Task{
    std::function<void()> func;
public:
    explicit FuncTask(auto f) : func(std::move(f)) {}
    void run() override {
        func();
    }
};

class ThreadPool {
public:
    static void WorkerThread(ThreadPool *master);
    void wait() const;

    explicit ThreadPool(size_t thread_count = 0);
    ~ThreadPool();

    void addTask(Task* task);
    Task* getTask();


private:
    std::atomic<bool>alive{true};
    std::atomic<uint32_t>pending_task_count{0};
    std::mutex lock;
    std::vector<std::thread> threads;
    std::list<Task *> tasks;
};

#endif //UNTITLED_THREAD_POOL_H