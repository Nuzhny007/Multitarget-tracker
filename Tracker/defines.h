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

///
/// \brief The PointObject class
///
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

    T Left() const
    {
        return m_pt.x;
    }
    T Top() const
    {
        return m_pt.y;
    }
    T Right() const
    {
        return m_pt.x;
    }
    T Bottom() const
    {
        return m_pt.y;
    }
    T Width() const
    {
        return 1;
    }
    T Height() const
    {
        return 1;
    }

    cv::Point_<T> Center() const
    {
        return m_pt;
    }

    cv::Rect_<T> BoundingRect() const
    {
        return cv::Rect_<T>(m_pt.x, m_pt.y, Width(), Height());
    }

    const value_type& Self() const
    {
        return m_pt;
    }

    void Points(std::vector<cv::Point>& points) const
    {
        points = { m_pt };
    }

private:
    value_type m_pt;
};

///
/// \brief The RectObject class
///
template<typename T>
class RectObject
{
public:
    typedef cv::Rect_<T> value_type;

    RectObject()
        : m_rect(0, 0, 0, 0)
    {

    }

    RectObject(const value_type& rect)
        : m_rect(rect)
    {

    }

    RectObject(const contour_t& contour)
    {
        m_rect = cv::boundingRect(contour);
    }

    T Left() const
    {
        return m_rect.x;
    }
    T Top() const
    {
        return m_rect.y;
    }
    T Right() const
    {
        return m_rect.x + m_rect.width;
    }
    T Bottom() const
    {
        return m_rect.y + m_rect.height;
    }
    T Width() const
    {
        return m_rect.width;
    }
    T Height() const
    {
        return m_rect.height;
    }

    cv::Point_<T> Center() const
    {
        return (m_rect.tl() + m_rect.br()) / 2;
    }

    cv::Rect_<T> BoundingRect() const
    {
        return m_rect;
    }

    const value_type& Self() const
    {
        return m_rect;
    }

    void Points(std::vector<cv::Point>& points) const
    {
        points = {
            m_rect.tl(), cv::Point(m_rect.x + m_rect.width, m_rect.y), m_rect.br(), cv::Point(m_rect.x, m_rect.y + m_rect.height)
        };
    }

private:
    value_type m_rect;
};

///
/// \brief The RectObject class
///
template<typename T>
class RotatedRectObject
{
public:
    typedef cv::RotatedRect value_type;

    RotatedRectObject()
        : m_rrect(cv::Point2f(0, 0), cv::Size2f(0, 0), 0)
    {

    }

    RotatedRectObject(const value_type& rrect)
        : m_rrect(rrect)
    {

    }

    RotatedRectObject(const contour_t& contour)
    {
        m_rrect = cv::minAreaRect(contour);
    }

    T Left() const
    {
        return m_rrect.boundingRect().x;
    }
    T Top() const
    {
        return m_rrect.boundingRect().y;
    }
    T Right() const
    {
        cv::Rect rect(m_rrect.boundingRect());
        return rect.x + rect.width;
    }
    T Bottom() const
    {
        cv::Rect rect(m_rrect.boundingRect());
        return rect.y + rect.height;
    }
    T Width() const
    {
        return m_rrect.boundingRect().width;
    }
    T Height() const
    {
        return m_rrect.boundingRect().height;
    }

    cv::Point_<T> Center() const
    {
        return m_rrect.center;
    }

    cv::Rect_<T> BoundingRect() const
    {
        return m_rrect.boundingRect();
    }

    const value_type& Self() const
    {
        return m_rrect;
    }

    void Points(std::vector<cv::Point>& points) const
    {
        cv::Point2f pts[4];
        m_rrect.points(pts);

        points = {
            cv::Point(pts[0]), cv::Point(pts[1]), cv::Point(pts[2]), cv::Point(pts[3])
        };
    }

private:
    value_type m_rrect;
};

///
/// \brief The CRegion class
///
template<typename OBJ>
class CRegion
{
public:
    typedef OBJ value_type;
    typedef value_type obj_type;

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

typedef CRegion<PointObject<int>> point_reg_t;
typedef std::vector<point_reg_t> point_regions_t;

typedef CRegion<RectObject<int>> rect_reg_t;
typedef std::vector<rect_reg_t> rect_regions_t;

typedef CRegion<RotatedRectObject<int>> rrect_reg_t;
typedef std::vector<rrect_reg_t> rrect_regions_t;
