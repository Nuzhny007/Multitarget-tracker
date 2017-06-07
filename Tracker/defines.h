#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
typedef float track_t;
typedef cv::Point_<track_t> Point_t;
#define Mat_t CV_32FC

typedef std::vector<cv::Point> contour_t;

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
template<typename T>
class PointObject
{
public:
    typedef cv::Point_<T> value_type;

    PointObject()
        : m_pt(0, 0)
    {

    }

    PointObject(const value_type& pt)
        : m_pt(pt)
    {

    }

    PointObject(const contour_t& contour)
    {
          cv::Moments mu = cv::moments(contour, false);
          m_pt.x = static_cast<T>(mu.m10 / mu.m00);
          m_pt.y = static_cast<T>(mu.m01 / mu.m00);
    }

    T GetLeft() const
    {
        return m_pt.x;
    }
    void SetLeft(T val)
    {
        m_pt.x = val;
    }

    T GetTop() const
    {
        return m_pt.y;
    }
    void SetTop(T val)
    {
        m_pt.y = val;
    }

    T GetRight() const
    {
        return m_pt.x;
    }
    void SetRight(T val)
    {
        m_pt.x = val;
    }

    T GetBottom() const
    {
        return m_pt.y;
    }
    void SetBottom(T val)
    {
        m_pt.y = val;
    }

    cv::Point_<T> Center() const
    {
        return m_pt;
    }

private:
    value_type m_pt;
};

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
template<typename T>
class RectObject
{
public:
    typedef cv::Rect_<T> value_type;

    RectObject()
        : m_rect(0, 0, 0, 0)
    {

    }

    RectObject(const value_type& pt)
        : m_rect(pt)
    {

    }

    RectObject(const contour_t& contour)
    {
        m_rect = cv::boundingRect(contour);
    }

    T GetLeft() const
    {
        return m_rect.x;
    }
    void SetLeft(T val)
    {
        m_rect.x = val;
    }

    T GetTop() const
    {
        return m_rect.y;
    }
    void SetTop(T val)
    {
        m_rect.y = val;
    }

    T GetRight() const
    {
        return m_rect.x + m_rect.width;
    }
    void SetRight(T val)
    {
        m_rect.width = val - m_rect.x;
    }

    T GetBottom() const
    {
        return m_rect.y + m_rect.height;
    }
    void SetBottom(T val)
    {
        m_rect.height = val - m_rect.y;
    }

    cv::Point_<T> Center() const
    {
        return (m_rect.tl() + m_rect.br()) / 2;
    }

private:
    value_type m_rect;
};

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
template<typename OBJ>
class CRegion
{
public:
    CRegion()
    {
    }

    CRegion(const OBJ& obj)
        : m_obj(obj)
    {
    }

    CRegion(const contour_t& contour)
        : m_obj(contour)
    {
    }

    OBJ m_obj;
    std::vector<cv::Point2f> m_points;
};

typedef std::vector<CRegion<PointObject<int>>> point_regions_t;
typedef std::vector<CRegion<RectObject<int>>> rect_regions_t;
