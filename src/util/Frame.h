#pragma once

#include <thread>

#include <opencv2/opencv.hpp>

///
/// \brief The Frame struct
///
class Frame
{
public:
    Frame() = default;
    Frame(cv::Mat imgBGR, bool useCLAHE)
    {
        m_mBGR = imgBGR;
        if (useCLAHE)
        {
            m_clahe = cv::createCLAHE(1.2, cv::Size(4, 4));
            AdjustMatBGR();
        }
    }

    ///
    void SetUseAdjust(bool useCLAHE)
    {
        if (useCLAHE)
        {
            m_clahe = cv::createCLAHE(1.2, cv::Size(4, 4));
            AdjustMatBGR();
        }
        else
        {
            m_clahe.reset();
        }
    }

    ///
    bool empty() const noexcept
    {
        return m_mBGR.empty();
    }

    ///
    const cv::Mat& GetMatBGR() const noexcept
    {
        return m_mBGR;
    }
    ///
    cv::Mat& GetMatBGRWrite()
    {
        m_umBGRGenerated = false;
        m_mGrayGenerated = false;
        m_umGrayGenerated = false;
        return m_mBGR;
    }
    ///
    bool AdjustMatBGR()
    {
        if (m_mBGR.empty() || m_clahe.empty())
            return false;

        cv::cvtColor(m_mBGR, m_mHSV, cv::COLOR_BGR2HSV);
        cv::split(m_mHSV, m_chansHSV);
        m_clahe->apply(m_chansHSV[2], m_chansHSV[2]);
        cv::merge(m_chansHSV, m_mHSV);
        cv::cvtColor(m_mHSV, m_mBGR, cv::COLOR_HSV2BGR);

        //std::cout << "AdjustMatBGR()" << std::endl;

        return true;
    }
    ///
    const cv::Mat& GetMatGray()
    {
        if (m_mGray.empty() || !m_mGrayGenerated)
        {
            if (m_umGray.empty() || !m_umGrayGenerated)
                cv::cvtColor(m_mBGR, m_mGray, cv::COLOR_BGR2GRAY);
            else
                m_mGray = m_umGray.getMat(cv::ACCESS_READ);
            m_mGrayGenerated = true;
        }
        return m_mGray;
    }
    ///
    const cv::UMat& GetUMatBGR()
    {
        std::thread::id lastThreadID = std::this_thread::get_id();

        if (m_umBGR.empty() || !m_umBGRGenerated || lastThreadID != m_umBGRThreadID)
        {
            m_umBGR = m_mBGR.getUMat(cv::ACCESS_READ);
            m_umBGRGenerated = true;
            m_umBGRThreadID = lastThreadID;
        }
        return m_umBGR;
    }
    ///
    const cv::UMat& GetUMatGray()
    {
        std::thread::id lastThreadID = std::this_thread::get_id();

        if (m_umGray.empty() || !m_umGrayGenerated || lastThreadID != m_umGrayThreadID)
        {
            if (m_mGray.empty() || !m_mGrayGenerated)
            {
                if (m_umBGR.empty() || !m_umBGRGenerated || lastThreadID != m_umBGRThreadID)
                    cv::cvtColor(m_mBGR, m_umGray, cv::COLOR_BGR2GRAY);
                else
                    cv::cvtColor(m_umBGR, m_umGray, cv::COLOR_BGR2GRAY);
            }
            else
            {
                m_umGray = m_mGray.getUMat(cv::ACCESS_READ);
            }
            m_umGrayGenerated = true;
            m_umGrayThreadID = lastThreadID;
        }
        return m_umGray;
    }

private:

#ifdef H14_TEST_ACCESS
    // Test-only hook: build/h14_test checks the thread guards of the UMat
    // caches (roadmap H14).
    friend class H14Test;
#endif

    cv::Mat m_mBGR;
    cv::Mat m_mGray;
    cv::UMat m_umBGR;
    cv::UMat m_umGray;
    bool m_umBGRGenerated = false;
    bool m_mGrayGenerated = false;
    bool m_umGrayGenerated = false;
    std::thread::id m_umBGRThreadID;
    std::thread::id m_umGrayThreadID;

    cv::Ptr<cv::CLAHE> m_clahe;
    cv::Mat m_mHSV;
    std::vector<cv::Mat> m_chansHSV;
};
