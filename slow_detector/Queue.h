#pragma once

#include <queue>
#include <list>
#include <vector>
#include <algorithm>
#include <mutex>
#include <condition_variable>

#define USE_DEQUE 1

///
/// A threadsafe-queue
///
template <class T>
class SafeQueue
{
public:
	///
	SafeQueue(void)
		: m_que()
		, m_mutex()
		, m_cond()
	{
	}

	///
	~SafeQueue(void)
	{
	}

	///
	/// Add an element to the queue
	///
	void enqueue(T t)
	{
		std::lock_guard<std::mutex> lock(m_mutex);
#if USE_DEQUE
        m_que.push(t);
#else
        m_que.insert(m_que.end(), t);
#endif
		m_cond.notify_one();
	}

    ///
    /// Add an element to the queue
    ///
    void enqueue(T t, size_t maxQueueSize)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
#if USE_DEQUE
        m_que.push(t);

        if (m_que.size() > maxQueueSize)
        {
            m_que.pop();
        }
#else
        m_que.insert(m_que.end(), t);

        if (m_que.size() > maxQueueSize)
        {
            m_que.erase(m_que.begin());
        }
#endif
        m_cond.notify_one();
    }

	///
	/// Add an element to the queue
	///
	template<typename RET_V, typename RET_F>
	RET_V enqueue(T t, RET_F && F)
	{
		std::lock_guard<std::mutex> lock(m_mutex);
#if USE_DEQUE
        m_que.push(t);
#else
        m_que.insert(m_que.end(), t);
#endif
		RET_V ret = F(m_que.front());
		m_cond.notify_one();

		return ret;
	}

	///
	/// Get the "front"-element
	/// If the queue is empty, wait till a element is avaiable
	///
	T dequeue(void)
	{
		std::unique_lock<std::mutex> lock(m_mutex);
		while (m_que.empty())
		{
			// release lock as long as the wait and reaquire it afterwards
			m_cond.wait(lock);
		}
		T val = m_que.front();
#if USE_DEQUE
        m_que.pop();
#else
        m_que.erase(m_que.begin());
#endif
		return val;
	}

	///
	size_t size()
	{
		std::unique_lock<std::mutex> lock(m_mutex);
		size_t res = m_que.size();
		return res;
	}

private:
#if USE_DEQUE
    std::queue<T> m_que;
#else
    std::list<T> m_que;
#endif
	mutable std::mutex m_mutex;
	std::condition_variable m_cond;
};
