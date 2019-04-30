#pragma once

#include <queue>
#include <deque>
#include <list>
#include <vector>
#include <algorithm>
#include <mutex>
#include <condition_variable>

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
	virtual ~SafeQueue(void)
	{
	}

protected:
    std::deque<T> m_que;
	mutable std::mutex m_mutex;
	std::condition_variable m_cond;

	///
	/// Add an element to the queue
	///
	void enqueue(T t)
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		m_que.push_back(t);
		m_cond.notify_one();
	}

	///
	/// Add an element to the queue
	///
	void enqueue(T t, size_t maxQueueSize)
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		m_que.push(t);

		if (m_que.size() > maxQueueSize)
		{
			m_que.pop();
		}
		m_cond.notify_one();
	}

	///
	/// Add an element to the queue
	///
	template<typename RET_V, typename RET_F>
	RET_V enqueue(T t, RET_F && F)
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		m_que.push(t);
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
		m_que.pop();

		return val;
	}

	///
	size_t size()
	{
		std::unique_lock<std::mutex> lock(m_mutex);
		size_t res = m_que.size();
		return res;
	}
};


struct FrameInfo;
///
/// A threadsafe-queue with Frames
///
class FramesQueue : public SafeQueue<FrameInfo>
{
public:
	void AddNewFrame(FrameInfo& frameInfo)
	{
		enqueue(frameInfo);
	}

	FrameInfo& GetLastUndetectedFrame()
	{
		std::unique_lock<std::mutex> lock(m_mutex);
		while (m_que.empty() || m_que.back().m_inDetector)
		{
			m_cond.wait(lock);
		}
		FrameInfo& frameInfo = m_que.back();
		frameInfo.m_inDetector = 1;
		return frameInfo;
	}

	FrameInfo& GetFirstDetectedFrame()
	{
		std::unique_lock<std::mutex> lock(m_mutex);
		auto SearchUntracked = [](const std::deque<FrameInfo>& que) -> int
		{
			int res = -1;
			for (int i = 0; i < que.size(); ++i)
			{
				if (que[i].m_inDetector != 1 && que[i].m_inTracker == 0)
				{
					res = i;
					break;
				}
			}
			return res;
		};
		while (SearchUntracked(m_que) == -1)
		{
			m_cond.wait(lock);
		}
		FrameInfo& frameInfo = m_que[SearchUntracked(m_que)];
		return frameInfo;
	}

	FrameInfo GetFirstProcessedFrame()
	{
		std::unique_lock<std::mutex> lock(m_mutex);
		while (m_que.empty() || m_que.front().m_inDetector)
		{
			m_cond.wait(lock);
		}
		FrameInfo frameInfo = m_que.front();
		m_que.pop_front();
		return frameInfo;
	}

	void Signal()
	{
		m_cond.notify_all();
	}

private:
};