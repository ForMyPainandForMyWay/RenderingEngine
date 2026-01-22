//
// Created by yyd on 2025/12/23.
//

#ifndef UNTITLED_THREAD_POOL_H
#define UNTITLED_THREAD_POOL_H
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
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
        // 1. 推导任务函数的返回值类型 (例如 int, double, void 等)
        // C++17 使用 std::invoke_result，C++11/14 使用 std::result_of
        using return_type = std::invoke_result_t<F, Args...>;

        // 2. 将函数和参数绑定，打包成一个 packaged_task
        // 这里的任务还是一个“函数”，但它执行后会把结果塞进 future 里
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        // 3. 获取 future，将来可以通过它拿到结果
        std::future<return_type> res = task->get_future();

        {
            std::lock_guard guard(lock);
            if (!alive) {
                // 如果线程池停止，返回一个空的 future 或者抛出异常
                // 这里简单处理，返回默认构造的 future (调用 get 会异常)
                return std::future<return_type>();
            }

            // 4. 关键步骤：把 packaged_task 包装进一个 void() lambda
            // 因为 tasks 队列只认 std::function<void()>
            // 我们利用 shared_ptr 的拷贝特性，让 lambda 捕获 task 指针
            tasks.emplace_back([task](){
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