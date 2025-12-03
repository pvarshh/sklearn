#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <iostream>

class ThreadPool {
public:
    static ThreadPool &instance() {
        static ThreadPool pool;
        return pool;
    }

    void parallel_for(int start, int end, std::function<void(int, int)> func) {
        int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0)
            num_threads = 4;

        int range = end - start;
        if (range <= 0) { return; }

        // Threshold for parallelism
        if (range < 64) {
            func(start, end);
            return;
        }

        int block_size = range / num_threads;
        std::atomic<int> tasks_remaining(num_threads - 1);

        for (int t = 0; t < num_threads - 1; ++t) {
            int t_start = start + t * block_size;
            int t_end = t_start + block_size;

            enqueue([=, &tasks_remaining, &func]()
                    { func(t_start, t_end);
                      tasks_remaining--; });
        }

        // Main thread does the last chunk
        func(start + (num_threads - 1) * block_size, end);

        // Wait for completion
        while (tasks_remaining > 0) { std::this_thread::yield(); }
    }

private:
    ThreadPool() {
        int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) { num_threads = 4; }
        stop = false;

        for (int i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] {
                while(true) {
                    std::function<void()> task; {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty()) { return; } 
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                } });
        }
    }

    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers) { worker.join(); }
    }

    void enqueue(std::function<void()> task) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.push(task);
        }
        condition.notify_one();
    }

    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

// Helper for parallel execution using the pool
static void parallel_for(int start, int end, std::function<void(int, int)> func) {
    ThreadPool::instance().parallel_for(start, end, func);
}

#endif // THREAD_POOL_H
