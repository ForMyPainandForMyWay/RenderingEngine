//
// Created by yyd on 2025/12/23.
//

#ifndef UNTITLED_THREAD_POOL_H
#define UNTITLED_THREAD_POOL_H
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <deque>
#include <functional>
#include <future>


class ThreadPool {
public:
    static void WorkerThread(ThreadPool *master);
    void wait() const;

    explicit ThreadPool(size_t thread_count = 0);
    ~ThreadPool();

    template<typename F, class... Args>
    auto addTask(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>
    {
        using return_type = std::invoke_result_t<F, Args...>;
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task->get_future();

        {
            std::lock_guard guard(lock);
            if (!alive) {
                return std::future<return_type>();
            }
            tasks.emplace_back([task]{
                (*task)(); // 执行 packaged_task，结果会自动进入 future
            });

            ++pending_task_count;
        }
        cv.notify_one();
        return res;
    }

private:
    std::atomic<bool>alive{true};
    std::atomic<uint32_t>pending_task_count{0};
    std::mutex lock;
    std::vector<std::thread> threads;
    std::deque<std::function<void()>> tasks;
    std::condition_variable cv;
    // std::unique_ptr<Task> getTask();
    bool popTask(std::function<void()>& task);
};

#endif //UNTITLED_THREAD_POOL_H