#include "Kalman.h"
#include <cmath>
#include <iostream>
#include <vector>

namespace
{
// RotatedRect orientation is undirected: rotating the rectangle by 180 degrees
// gives the same rectangle. The tracker therefore works with angles modulo 180
// degrees, otherwise a detector convention change (or crossing the +-90 degree
// boundary) looks like a huge rotation to the linear Kalman filter.
constexpr track_t kHalfTurnDeg = static_cast<track_t>(180);

// Bring an angle into (-90, 90] degrees.
track_t WrapHalfTurnDeg(track_t angle)
{
    angle = std::fmod(angle, kHalfTurnDeg);
    if (angle <= -90)
        angle += kHalfTurnDeg;
    else if (angle > 90)
        angle -= kHalfTurnDeg;
    return angle;
}

// Shift `angle` by multiples of 180 degrees so that it is as close as possible
// to `reference`. Used to unroll the angle measurement before the linear
// Kalman correction.
track_t UnwrapHalfTurnDeg(track_t angle, track_t reference)
{
    return reference + WrapHalfTurnDeg(angle - reference);
}

// Circular mean of undirected orientations (period 180 degrees).
track_t CircularMeanHalfTurnDeg(const std::vector<cv::RotatedRect>& rects)
{
    double sumSin = 0.0;
    double sumCos = 0.0;
    for (const auto& rect : rects)
    {
        const double rad = static_cast<double>(rect.angle) * CV_PI / 180.0;
        sumSin += std::sin(2.0 * rad);
        sumCos += std::cos(2.0 * rad);
    }
    return static_cast<track_t>(0.5 * std::atan2(sumSin, sumCos) * 180.0 / CV_PI);
}

// Constant-acceleration transition for `dims` independent coordinates:
// state = [p0..pN-1, v0..vN-1, a0..aN-1],
// p' = p + dt * v + dt^2 / 2 * a, v' = v + dt * a, a' = a.
cv::Mat MakeAccelerationTransition(int dims, track_t dt)
{
    const int size = dims * 3;
    const track_t halfDt2 = static_cast<track_t>(0.5) * dt * dt;
    cv::Mat transition = cv::Mat::eye(size, size, Mat_t(1));
    for (int i = 0; i < dims; ++i)
    {
        transition.at<track_t>(i, dims + i) = dt;
        transition.at<track_t>(i, 2 * dims + i) = halfDt2;
        transition.at<track_t>(dims + i, 2 * dims + i) = dt;
    }
    return transition;
}

// Process noise for the constant-acceleration model with white acceleration
// noise: G = [dt^2 / 2, dt, 1]. The previous implementation reused the CV
// acceleration noise for the acceleration state (dt^3/2 and dt^2 couplings),
// which was inconsistent with the acceleration transition (roadmap H6).
cv::Mat MakeAccelerationProcessNoise(int dims, track_t dt, track_t accelNoiseMag)
{
    const int size = dims * 3;
    const track_t dt2 = dt * dt;
    const track_t dt3 = dt2 * dt;
    const track_t dt4 = dt3 * dt;
    const track_t q11 = dt4 / static_cast<track_t>(4);
    const track_t q12 = dt3 / static_cast<track_t>(2);
    const track_t q13 = dt2 / static_cast<track_t>(2);
    const track_t q22 = dt2;
    const track_t q23 = dt;
    const track_t q33 = static_cast<track_t>(1);

    cv::Mat noise = cv::Mat::zeros(size, size, Mat_t(1));
    for (int i = 0; i < dims; ++i)
    {
        noise.at<track_t>(i, i) = q11;
        noise.at<track_t>(i, dims + i) = q12;
        noise.at<track_t>(dims + i, i) = q12;
        noise.at<track_t>(i, 2 * dims + i) = q13;
        noise.at<track_t>(2 * dims + i, i) = q13;
        noise.at<track_t>(dims + i, dims + i) = q22;
        noise.at<track_t>(dims + i, 2 * dims + i) = q23;
        noise.at<track_t>(2 * dims + i, dims + i) = q23;
        noise.at<track_t>(2 * dims + i, 2 * dims + i) = q33;
    }
    noise *= accelNoiseMag;
    return noise;
}

// Keep the orientation of a RotatedRect in the canonical (-90, 90] range.
// A 180 degree rotation maps a rectangle onto itself, so the size is untouched.
cv::RotatedRect NormalizeRotatedRectAngle(const cv::RotatedRect& rect)
{
    cv::RotatedRect result = rect;
    result.angle = WrapHalfTurnDeg(result.angle);
    return result;
}
} // namespace

#ifdef USE_OCV_UKF
#if (((CV_VERSION_MAJOR == 4) && (CV_VERSION_MINOR < 5)) || ((CV_VERSION_MAJOR == 4) && (CV_VERSION_MINOR == 5) && (CV_VERSION_REVISION < 1)) || (CV_VERSION_MAJOR == 3))
namespace kalman = cv::tracking;
#else
namespace kalman = cv::detail::tracking;
#endif
#endif

///
/// \brief TKalmanFilter::TKalmanFilter
/// \param type
/// \param useAcceleration
/// \param deltaTime
/// \param accelNoiseMag
///
TKalmanFilter::TKalmanFilter(tracking::KalmanType type,
                             bool useAcceleration,
                             track_t deltaTime, // time increment (lower values makes target more "massive")
                             track_t accelNoiseMag)
    :
      m_accelNoiseMag(accelNoiseMag),
      m_deltaTime(deltaTime),
      m_deltaTimeMin(deltaTime),
      m_deltaTimeMax(2 * deltaTime),
      m_type(type),
      m_useAcceleration(useAcceleration)
{
    m_deltaStep = (m_deltaTimeMax - m_deltaTimeMin) / m_deltaStepsCount;
}

///
/// \brief TKalmanFilter::CreateLinear
/// \param xy0
/// \param xyv0
///
void TKalmanFilter::CreateLinear(Point_t xy0, Point_t xyv0)
{
    // We don't know acceleration, so, assume it to process noise.
    // But we can guess, the range of acceleration values thich can be achieved by tracked object.
    // Process noise. (standard deviation of acceleration: m/s^2)
    // shows, woh much target can accelerate.

    // 4 state variables, 2 measurements
    m_velocityXInd = 2; // vx
    m_velocityYInd = 3; // vy
    m_linearKalman.init(4, 2, 0, El_t);
    // Transition cv::Matrix
    m_linearKalman.transitionMatrix = cv::Mat_<track_t>({ 4, 4 },
                                      { 1, 0, m_deltaTime, 0,
                                        0, 1, 0, m_deltaTime,
                                        0, 0, 1, 0,
                                        0, 0, 0, 1 });

    // init...
    m_lastPointResult = xy0;
    m_linearKalman.statePre.at<track_t>(0) = xy0.x;  // x
    m_linearKalman.statePre.at<track_t>(1) = xy0.y;  // y
    m_linearKalman.statePre.at<track_t>(2) = xyv0.x; // vx
    m_linearKalman.statePre.at<track_t>(3) = xyv0.y; // vy

    m_linearKalman.statePost.at<track_t>(0) = xy0.x;
    m_linearKalman.statePost.at<track_t>(1) = xy0.y;
    m_linearKalman.statePost.at<track_t>(2) = xyv0.x;
    m_linearKalman.statePost.at<track_t>(3) = xyv0.y;

    cv::setIdentity(m_linearKalman.measurementMatrix);

    m_linearKalman.processNoiseCov = cv::Mat_<track_t>({ 4, 4 },
                                     { pow(m_deltaTime, static_cast<track_t>(4.0)) / static_cast<track_t>(4.0), 0, pow(m_deltaTime, static_cast<track_t>(3.0)) / static_cast<track_t>(2.0), 0,
                                       0, pow(m_deltaTime, static_cast<track_t>(4.0)) / static_cast<track_t>(4.0), 0, pow(m_deltaTime, static_cast<track_t>(3.0)) / static_cast<track_t>(2.0),
                                       pow(m_deltaTime, static_cast<track_t>(3.0)) / static_cast<track_t>(2.0), 0, pow(m_deltaTime, static_cast<track_t>(2.0)), 0,
                                       0, pow(m_deltaTime, static_cast<track_t>(3.0)) / static_cast<track_t>(2.0), 0, pow(m_deltaTime, static_cast<track_t>(2.0)) });


    m_linearKalman.processNoiseCov *= m_accelNoiseMag;

    cv::setIdentity(m_linearKalman.measurementNoiseCov, cv::Scalar::all(0.1));

    cv::setIdentity(m_linearKalman.errorCovPost, cv::Scalar::all(.1));

    m_initialPoints.reserve(MIN_INIT_VALS);

    m_initialized = true;
}

///
/// \brief TKalmanFilter::CreateLinear
/// \param rect0
/// \param rectv0
///
void TKalmanFilter::CreateLinear(cv::Rect_<track_t> rect0, Point_t rectv0)
{
    // We don't know acceleration, so, assume it to process noise.
    // But we can guess, the range of acceleration values thich can be achieved by tracked object.
    // Process noise. (standard deviation of acceleration: m/s^2)
    // shows, woh much target can accelerate.

    // 8 state variables (x, y, width, height, vx, vy, vw, vh), 4 measurements (x, y, width, height)
    m_velocityXInd = 4; // vx
    m_velocityYInd = 5; // vy
    m_linearKalman.init(8, 4, 0, El_t);
    // Transition cv::Matrix
    m_linearKalman.transitionMatrix = cv::Mat_<track_t>({ 8, 8 },
        { 1, 0, 0, 0, m_deltaTime, 0,           0,           0,
                                        0, 1, 0, 0, 0,           m_deltaTime, 0,           0,
                                        0, 0, 1, 0, 0,           0,           m_deltaTime, 0,
                                        0, 0, 0, 1, 0,           0,           0,           m_deltaTime,
                                        0, 0, 0, 0, 1,           0,           0,           0,
                                        0, 0, 0, 0, 0,           1,           0,           0,
                                        0, 0, 0, 0, 0,           0,           1,           0,
                                        0, 0, 0, 0, 0,           0,           0,           1 });

    // init...
    m_linearKalman.statePre.at<track_t>(0) = rect0.x;      // x
    m_linearKalman.statePre.at<track_t>(1) = rect0.y;      // y
    m_linearKalman.statePre.at<track_t>(2) = rect0.width;  // width
    m_linearKalman.statePre.at<track_t>(3) = rect0.height; // height
    m_linearKalman.statePre.at<track_t>(4) = rectv0.x;     // dx
    m_linearKalman.statePre.at<track_t>(5) = rectv0.y;     // dy
    m_linearKalman.statePre.at<track_t>(6) = 0;            // dw
    m_linearKalman.statePre.at<track_t>(7) = 0;            // dh

    m_linearKalman.statePost.at<track_t>(0) = rect0.x;
    m_linearKalman.statePost.at<track_t>(1) = rect0.y;
    m_linearKalman.statePost.at<track_t>(2) = rect0.width;
    m_linearKalman.statePost.at<track_t>(3) = rect0.height;
    m_linearKalman.statePost.at<track_t>(4) = rectv0.x;
    m_linearKalman.statePost.at<track_t>(5) = rectv0.y;
    m_linearKalman.statePost.at<track_t>(6) = 0;
    m_linearKalman.statePost.at<track_t>(7) = 0;

    cv::setIdentity(m_linearKalman.measurementMatrix);

    track_t n1 = pow(m_deltaTime, 4.f) / 4.f;
    track_t n2 = pow(m_deltaTime, 3.f) / 2.f;
    track_t n3 = pow(m_deltaTime, 2.f);
    m_linearKalman.processNoiseCov = cv::Mat_<track_t>({ 8, 8 },
                                     { n1, 0,  0,  0,  n2, 0,  0,  0,
                                       0,  n1, 0,  0,  0,  n2, 0,  0,
                                       0,  0,  n1, 0,  0,  0,  n2, 0,
                                       0,  0,  0,  n1, 0,  0,  0,  n2,
                                       n2, 0,  0,  0,  n3, 0,  0,  0,
                                       0,  n2, 0,  0,  0,  n3, 0,  0,
                                       0,  0,  n2, 0,  0,  0,  n3, 0,
                                       0,  0,  0,  n2, 0,  0,  0,  n3 });

    m_linearKalman.processNoiseCov *= m_accelNoiseMag;

    cv::setIdentity(m_linearKalman.measurementNoiseCov, cv::Scalar::all(0.1));

    cv::setIdentity(m_linearKalman.errorCovPost, cv::Scalar::all(.1));

    m_initialRects.reserve(MIN_INIT_VALS);

    m_initialized = true;
}

///
/// \brief TKalmanFilter::CreateLinear
/// \param rrect0
/// \param rrectv0
///
void TKalmanFilter::CreateLinear(cv::RotatedRect rrect0, Point_t rrectv0)
{
	// We don't know acceleration, so, assume it to process noise.
	// But we can guess, the range of acceleration values thich can be achieved by tracked object.
	// Process noise. (standard deviation of acceleration: m/s^2)
	// shows, woh much target can accelerate.

	// 10 state variables (x, y, width, height, angle, vx, vy, vw, vh, vangle), 5 measurements (x, y, width, height, angle)
	m_velocityXInd = 5; // vx
	m_velocityYInd = 6; // vy
	m_linearKalman.init(10, 5, 0, El_t);
	// Transition cv::Matrix
    m_linearKalman.transitionMatrix = cv::Mat_<track_t>({ 10, 10 },
                                     { 1, 0, 0, 0, 0, m_deltaTime, 0,           0,           0,           0,
                                       0, 1, 0, 0, 0, 0,           m_deltaTime, 0,           0,           0,
                                       0, 0, 1, 0, 0, 0,           0,           m_deltaTime, 0,           0,
                                       0, 0, 0, 1, 0, 0,           0,           0,           m_deltaTime, 0,
                                       0, 0, 0, 0, 1, 0,           0,           0,           0,           m_deltaTime,
                                       0, 0, 0, 0, 0, 1,           0,           0,           0,           0,
                                       0, 0, 0, 0, 0, 0,           1,           0,           0,           0,
                                       0, 0, 0, 0, 0, 0,           0,           1,           0,           0,
                                       0, 0, 0, 0, 0, 0,           0,           0,           1,           0,
                                       0, 0, 0, 0, 0, 0,           0,           0,           0,           1 });
	// init...
	m_linearKalman.statePre.at<track_t>(0) = rrect0.center.x;      // x
	m_linearKalman.statePre.at<track_t>(1) = rrect0.center.y;      // y
	m_linearKalman.statePre.at<track_t>(2) = rrect0.size.width;    // width
	m_linearKalman.statePre.at<track_t>(3) = rrect0.size.height;   // height
	m_linearKalman.statePre.at<track_t>(4) = rrect0.angle;         // angle
	m_linearKalman.statePre.at<track_t>(5) = rrectv0.x;            // dx
	m_linearKalman.statePre.at<track_t>(6) = rrectv0.y;            // dy
	m_linearKalman.statePre.at<track_t>(7) = 0;                    // dw
	m_linearKalman.statePre.at<track_t>(8) = 0;                    // dh
	m_linearKalman.statePre.at<track_t>(9) = 0;                    // da

	m_linearKalman.statePost.at<track_t>(0) = rrect0.center.x;
	m_linearKalman.statePost.at<track_t>(1) = rrect0.center.y;
	m_linearKalman.statePost.at<track_t>(2) = rrect0.size.width;
	m_linearKalman.statePost.at<track_t>(3) = rrect0.size.height;
	m_linearKalman.statePost.at<track_t>(4) = rrect0.angle;
	m_linearKalman.statePost.at<track_t>(5) = rrectv0.x;
	m_linearKalman.statePost.at<track_t>(6) = rrectv0.y;
	m_linearKalman.statePost.at<track_t>(7) = 0;
	m_linearKalman.statePost.at<track_t>(8) = 0;
	m_linearKalman.statePost.at<track_t>(9) = 0;

	cv::setIdentity(m_linearKalman.measurementMatrix);

	track_t n1 = pow(m_deltaTime, 4.f) / 4.f;
	track_t n2 = pow(m_deltaTime, 3.f) / 2.f;
	track_t n3 = pow(m_deltaTime, 2.f);
    m_linearKalman.processNoiseCov = cv::Mat_<track_t>({ 10, 10 },
        { n1, 0,  0,  0,  0,  n2, 0,  0,  0,  0,
        0,  n1, 0,  0,  0,  0,  n2, 0,  0,  0,
        0,  0,  n1, 0,  0,  0,  0,  n2, 0,  0,
        0,  0,  0,  n1, 0,  0,  0,  0,  n2, 0,
        0,  0,  0,  0,  n1, 0,  0,  0,  0,  n2,
        n2, 0,  0,  0,  0,  n3, 0,  0,  0,	0,
        0,  n2, 0,  0,  0,  0,  n3, 0,  0,	0,
        0,  0,  n2, 0,  0,  0,  0,  n3, 0,	0,
        0,  0,  0,  n2, 0,  0,  0,  0,  n3, 0,
        0,  0,  0,  0,  n2, 0,  0,  0,  0,  n3 });

	m_linearKalman.processNoiseCov *= m_accelNoiseMag;

	cv::setIdentity(m_linearKalman.measurementNoiseCov, cv::Scalar::all(0.1));

	cv::setIdentity(m_linearKalman.errorCovPost, cv::Scalar::all(.1));

	m_initialRRects.reserve(MIN_INIT_VALS);

	m_initialized = true;
}

///
/// \brief TKalmanFilter::CreateLinearAcceleration
/// \param xy0
/// \param xyv0
///
void TKalmanFilter::CreateLinearAcceleration(Point_t xy0, Point_t xyv0)
{
    // 6 state variables (x, y, vx, vy, ax, ay), 2 measurements
    m_velocityXInd = 2; // vx
    m_velocityYInd = 3; // vy
    m_linearKalman.init(6, 2, 0, El_t);
    // Transition cv::Matrix
    const track_t dt = m_deltaTime;
    const track_t dt2 = 0.5f * m_deltaTime * m_deltaTime;
    m_linearKalman.transitionMatrix = cv::Mat_<track_t>({ 6, 6 },
                                     { 1, 0, dt, 0,  dt2, 0,
                                       0, 1, 0,  dt, 0,   dt2,
                                       0, 0, 1,  0,  dt,  0,
                                       0, 0, 0,  1,  0,   dt,
                                       0, 0, 0,  0,  1,   0,
                                       0, 0, 0,  0,  0,   1 });

    // init...
    m_lastPointResult = xy0;
    m_linearKalman.statePre.at<track_t>(0) = xy0.x;  // x
    m_linearKalman.statePre.at<track_t>(1) = xy0.y;  // y
    m_linearKalman.statePre.at<track_t>(2) = xyv0.x; // vx
    m_linearKalman.statePre.at<track_t>(3) = xyv0.y; // vy
    m_linearKalman.statePre.at<track_t>(4) = 0;      // ax
    m_linearKalman.statePre.at<track_t>(5) = 0;      // ay

    m_linearKalman.statePost.at<track_t>(0) = xy0.x;
    m_linearKalman.statePost.at<track_t>(1) = xy0.y;
    m_linearKalman.statePost.at<track_t>(2) = xyv0.x;
    m_linearKalman.statePost.at<track_t>(3) = xyv0.y;
    m_linearKalman.statePost.at<track_t>(4) = 0;
    m_linearKalman.statePost.at<track_t>(5) = 0;

    cv::setIdentity(m_linearKalman.measurementMatrix);

    m_linearKalman.processNoiseCov = MakeAccelerationProcessNoise(2, m_deltaTime, m_accelNoiseMag);

    cv::setIdentity(m_linearKalman.measurementNoiseCov, cv::Scalar::all(0.1));

    cv::setIdentity(m_linearKalman.errorCovPost, cv::Scalar::all(.1));

    m_initialPoints.reserve(MIN_INIT_VALS);

    m_initialized = true;
}

///
/// \brief TKalmanFilter::CreateLinearAcceleration
/// \param rect0
/// \param rectv0
///
void TKalmanFilter::CreateLinearAcceleration(cv::Rect_<track_t> rect0, Point_t rectv0)
{
    // 12 state variables (x, y, width, height, vx, vy, vw, vh, ax, ay, aw, ah), 4 measurements (x, y, width, height)
    m_velocityXInd = 4; // vx
    m_velocityYInd = 5; // vy
    m_linearKalman.init(12, 4, 0, El_t);
    // Transition cv::Matrix
    const track_t dt = m_deltaTime;
    const track_t dt2 = 0.5f * m_deltaTime * m_deltaTime;
    m_linearKalman.transitionMatrix = cv::Mat_<track_t>({ 12, 12 },
                                     { 1, 0, 0, 0, dt, 0,  0,  0,  dt2, 0,   0,   0,
                                       0, 1, 0, 0, 0,  dt, 0,  0,  0,   dt2, 0,   0,
                                       0, 0, 1, 0, 0,  0,  dt, 0,  0,   0,   dt2, 0,
                                       0, 0, 0, 1, 0,  0,  0,  dt, 0,   0,   0,   dt2,
                                       0, 0, 0, 0, 1,  0,  0,  0,  dt,  0,   0,   0,
                                       0, 0, 0, 0, 0,  1,  0,  0,  0,   dt,  0,   0,
                                       0, 0, 0, 0, 0,  0,  1,  0,  0,   0,   dt,  0,
                                       0, 0, 0, 0, 0,  0,  0,  1,  0,   0,   0,   dt,
                                       0, 0, 0, 0, 0,  0,  0,  0,  1,   0,   0,   0,
                                       0, 0, 0, 0, 0,  0,  0,  0,  0,   1,   0,   0,
                                       0, 0, 0, 0, 0,  0,  0,  0,  0,   0,   1,   0,
                                       0, 0, 0, 0, 0,  0,  0,  0,  0,   0,   0,   1 });

    // init...
    m_linearKalman.statePre.at<track_t>(0) = rect0.x;      // x
    m_linearKalman.statePre.at<track_t>(1) = rect0.y;      // y
    m_linearKalman.statePre.at<track_t>(2) = rect0.width;  // width
    m_linearKalman.statePre.at<track_t>(3) = rect0.height; // height
    m_linearKalman.statePre.at<track_t>(4) = rectv0.x;     // dx
    m_linearKalman.statePre.at<track_t>(5) = rectv0.y;     // dy
    m_linearKalman.statePre.at<track_t>(6) = 0;            // dw
    m_linearKalman.statePre.at<track_t>(7) = 0;            // dh
    m_linearKalman.statePre.at<track_t>(8) = 0;            // ax
    m_linearKalman.statePre.at<track_t>(9) = 0;            // ay
    m_linearKalman.statePre.at<track_t>(10) = 0;           // aw
    m_linearKalman.statePre.at<track_t>(11) = 0;           // ah

    m_linearKalman.statePost.at<track_t>(0) = rect0.x;
    m_linearKalman.statePost.at<track_t>(1) = rect0.y;
    m_linearKalman.statePost.at<track_t>(2) = rect0.width;
    m_linearKalman.statePost.at<track_t>(3) = rect0.height;
    m_linearKalman.statePost.at<track_t>(4) = rectv0.x;
    m_linearKalman.statePost.at<track_t>(5) = rectv0.y;
    m_linearKalman.statePost.at<track_t>(6) = 0;
    m_linearKalman.statePost.at<track_t>(7) = 0;
    m_linearKalman.statePost.at<track_t>(8) = 0;
    m_linearKalman.statePost.at<track_t>(9) = 0;
    m_linearKalman.statePost.at<track_t>(10) = 0;
    m_linearKalman.statePost.at<track_t>(11) = 0;

    cv::setIdentity(m_linearKalman.measurementMatrix);

    m_linearKalman.processNoiseCov = MakeAccelerationProcessNoise(4, m_deltaTime, m_accelNoiseMag);

    cv::setIdentity(m_linearKalman.measurementNoiseCov, cv::Scalar::all(0.1));

    cv::setIdentity(m_linearKalman.errorCovPost, cv::Scalar::all(.1));

    m_initialRects.reserve(MIN_INIT_VALS);

    m_initialized = true;
}

///
/// \brief TKalmanFilter::CreateLinearAcceleration
/// \param rect0
/// \param rectv0
///
void TKalmanFilter::CreateLinearAcceleration(cv::RotatedRect rrect0, Point_t rrectv0)
{
    // 15 state variables (x, y, width, height, angle,
    //                      vx, vy, vw, vh, vangle,
    //                      ax, ay, aw, ah, aangle),
    // 5 measurements (x, y, width, height, angle).
    m_velocityXInd = 5; // vx
    m_velocityYInd = 6; // vy
    m_linearKalman.init(15, 5, 0, El_t);
    m_linearKalman.transitionMatrix = MakeAccelerationTransition(5, m_deltaTime);

    // init...
    m_lastRRectResult = rrect0;
    m_linearKalman.statePre.at<track_t>(0) = rrect0.center.x;    // x
    m_linearKalman.statePre.at<track_t>(1) = rrect0.center.y;    // y
    m_linearKalman.statePre.at<track_t>(2) = rrect0.size.width;  // width
    m_linearKalman.statePre.at<track_t>(3) = rrect0.size.height; // height
    m_linearKalman.statePre.at<track_t>(4) = rrect0.angle;       // angle
    m_linearKalman.statePre.at<track_t>(5) = rrectv0.x;          // vx
    m_linearKalman.statePre.at<track_t>(6) = rrectv0.y;          // vy
    m_linearKalman.statePre.at<track_t>(7) = 0;                  // vw
    m_linearKalman.statePre.at<track_t>(8) = 0;                  // vh
    m_linearKalman.statePre.at<track_t>(9) = 0;                  // vangle
    m_linearKalman.statePre.at<track_t>(10) = 0;                 // ax
    m_linearKalman.statePre.at<track_t>(11) = 0;                 // ay
    m_linearKalman.statePre.at<track_t>(12) = 0;                 // aw
    m_linearKalman.statePre.at<track_t>(13) = 0;                 // ah
    m_linearKalman.statePre.at<track_t>(14) = 0;                 // aangle

    m_linearKalman.statePost.at<track_t>(0) = rrect0.center.x;
    m_linearKalman.statePost.at<track_t>(1) = rrect0.center.y;
    m_linearKalman.statePost.at<track_t>(2) = rrect0.size.width;
    m_linearKalman.statePost.at<track_t>(3) = rrect0.size.height;
    m_linearKalman.statePost.at<track_t>(4) = rrect0.angle;
    m_linearKalman.statePost.at<track_t>(5) = rrectv0.x;
    m_linearKalman.statePost.at<track_t>(6) = rrectv0.y;
    m_linearKalman.statePost.at<track_t>(7) = 0;
    m_linearKalman.statePost.at<track_t>(8) = 0;
    m_linearKalman.statePost.at<track_t>(9) = 0;
    m_linearKalman.statePost.at<track_t>(10) = 0;
    m_linearKalman.statePost.at<track_t>(11) = 0;
    m_linearKalman.statePost.at<track_t>(12) = 0;
    m_linearKalman.statePost.at<track_t>(13) = 0;
    m_linearKalman.statePost.at<track_t>(14) = 0;

    cv::setIdentity(m_linearKalman.measurementMatrix);

    m_linearKalman.processNoiseCov = MakeAccelerationProcessNoise(5, m_deltaTime, m_accelNoiseMag);

    cv::setIdentity(m_linearKalman.measurementNoiseCov, cv::Scalar::all(0.1));

    cv::setIdentity(m_linearKalman.errorCovPost, cv::Scalar::all(.1));

    m_initialRRects.reserve(MIN_INIT_VALS);

    m_initialized = true;
}

#ifdef USE_OCV_UKF
//---------------------------------------------------------------------------
enum class UkfModelKind
{
    Point, // state: x, y, vx, vy, ax, ay
    Rect,  // state: x, y, vx, vy, ax, ay, width, height
    RRect  // state: x, y, vx, vy, ax, ay, width, height, angle, vangle
};

class AcceleratedModel final : public kalman::UkfSystemModel
{
public:
    AcceleratedModel(track_t deltaTime, UkfModelKind kind)
        :
          kalman::UkfSystemModel(),
          m_deltaTime(deltaTime),
          m_kind(kind)
    {
    }

    void stateConversionFunction(const cv::Mat& x_k, const cv::Mat& u_k, const cv::Mat& v_k, cv::Mat& x_kplus1) override
    {
        const track_t dt = m_deltaTime;
        const track_t halfDt2 = static_cast<track_t>(0.5) * dt * dt;
        const track_t x0 = x_k.at<track_t>(0, 0);
        const track_t y0 = x_k.at<track_t>(1, 0);
        const track_t vx0 = x_k.at<track_t>(2, 0);
        const track_t vy0 = x_k.at<track_t>(3, 0);
        const track_t ax0 = x_k.at<track_t>(4, 0);
        const track_t ay0 = x_k.at<track_t>(5, 0);

        x_kplus1.at<track_t>(0, 0) = x0 + vx0 * dt + ax0 * halfDt2;
        x_kplus1.at<track_t>(1, 0) = y0 + vy0 * dt + ay0 * halfDt2;
        x_kplus1.at<track_t>(2, 0) = vx0 + ax0 * dt;
        x_kplus1.at<track_t>(3, 0) = vy0 + ay0 * dt;
        x_kplus1.at<track_t>(4, 0) = ax0;
        x_kplus1.at<track_t>(5, 0) = ay0;

        if (m_kind == UkfModelKind::Rect)
        {
            x_kplus1.at<track_t>(6, 0) = x_k.at<track_t>(6, 0); // width
            x_kplus1.at<track_t>(7, 0) = x_k.at<track_t>(7, 0); // height
        }
        else if (m_kind == UkfModelKind::RRect)
        {
            x_kplus1.at<track_t>(6, 0) = x_k.at<track_t>(6, 0); // width
            x_kplus1.at<track_t>(7, 0) = x_k.at<track_t>(7, 0); // height
            x_kplus1.at<track_t>(8, 0) = x_k.at<track_t>(8, 0) + x_k.at<track_t>(9, 0) * dt; // angle
            x_kplus1.at<track_t>(9, 0) = x_k.at<track_t>(9, 0); // angular velocity
        }

        if (v_k.size() == u_k.size())
            x_kplus1 += v_k + u_k;
        else
            x_kplus1 += v_k;
    }

    void measurementFunction(const cv::Mat& x_k, const cv::Mat& n_k, cv::Mat& z_k) override
    {
        // h(x) maps the state to the measured components. It must NOT advance
        // the state by dt: predict() has already done that and correct() applies
        // h to the predicted state. The previous version added a second dt,
        // so the filter corrected against a position one step ahead of the
        // measurement.
        z_k.at<track_t>(0, 0) = x_k.at<track_t>(0, 0) + n_k.at<track_t>(0, 0);
        z_k.at<track_t>(1, 0) = x_k.at<track_t>(1, 0) + n_k.at<track_t>(1, 0);

        if (m_kind == UkfModelKind::Rect)
        {
            z_k.at<track_t>(2, 0) = x_k.at<track_t>(6, 0) + n_k.at<track_t>(2, 0);
            z_k.at<track_t>(3, 0) = x_k.at<track_t>(7, 0) + n_k.at<track_t>(3, 0);
        }
        else if (m_kind == UkfModelKind::RRect)
        {
            z_k.at<track_t>(2, 0) = x_k.at<track_t>(6, 0) + n_k.at<track_t>(2, 0); // width
            z_k.at<track_t>(3, 0) = x_k.at<track_t>(7, 0) + n_k.at<track_t>(3, 0); // height
            z_k.at<track_t>(4, 0) = x_k.at<track_t>(8, 0) + n_k.at<track_t>(4, 0); // angle
        }
    }

private:
    track_t m_deltaTime;
    UkfModelKind m_kind;
};

//---------------------------------------------------------------------------
void TKalmanFilter::CreateUnscented(Point_t xy0, Point_t xyv0)
{
    // State layout: x, y, vx, vy, ax, ay
    m_velocityXInd = 2; // vx
    m_velocityYInd = 3; // vy
    int MP = 2;
    int DP = 6;
    int CP = 0;

    cv::Mat processNoiseCov = cv::Mat::zeros(DP, DP, Mat_t(1));
    for (int i = 0; i < DP; ++i)
        processNoiseCov.at<track_t>(i, i) = 1e-3f;

    cv::Mat measurementNoiseCov = cv::Mat::zeros(MP, MP, Mat_t(1));
    measurementNoiseCov.at<track_t>(0, 0) = 1e-6f;
    measurementNoiseCov.at<track_t>(1, 1) = 1e-6f;

    cv::Mat initState(DP, 1, Mat_t(1));
    initState.at<track_t>(0, 0) = xy0.x;
    initState.at<track_t>(1, 0) = xy0.y;
    initState.at<track_t>(2, 0) = xyv0.x;
    initState.at<track_t>(3, 0) = xyv0.y;
    initState.at<track_t>(4, 0) = 0;
    initState.at<track_t>(5, 0) = 0;

    cv::Mat P = 1e-6f * cv::Mat::eye(DP, DP, Mat_t(1));

    cv::Ptr<AcceleratedModel> model(new AcceleratedModel(m_deltaTime, UkfModelKind::Point));
    kalman::UnscentedKalmanFilterParams params(DP, MP, CP, 0, 0, model);
    params.dataType = Mat_t(1);
    params.stateInit = initState.clone();
    params.errorCovInit = P.clone();
    params.measurementNoiseCov = measurementNoiseCov.clone();
    params.processNoiseCov = processNoiseCov.clone();

    params.alpha = 1.0;
    params.beta = 2.0;
    params.k = -2.0;

    m_uncsentedKalman = kalman::createUnscentedKalmanFilter(params);

    m_initialized = true;
}

//---------------------------------------------------------------------------
void TKalmanFilter::CreateUnscented(cv::Rect_<track_t> rect0, Point_t rectv0)
{
    // State layout: x, y, vx, vy, ax, ay, width, height
    m_velocityXInd = 2; // vx
    m_velocityYInd = 3; // vy
    int MP = 4;
    int DP = 8;
    int CP = 0;

    cv::Mat processNoiseCov = cv::Mat::zeros(DP, DP, Mat_t(1));
    processNoiseCov.at<track_t>(0, 0) = 1e-3f;
    processNoiseCov.at<track_t>(1, 1) = 1e-3f;
    processNoiseCov.at<track_t>(2, 2) = 1e-3f;
    processNoiseCov.at<track_t>(3, 3) = 1e-3f;
    processNoiseCov.at<track_t>(4, 4) = 1e-3f;
    processNoiseCov.at<track_t>(5, 5) = 1e-3f;
    processNoiseCov.at<track_t>(6, 6) = 1e-3f;
    processNoiseCov.at<track_t>(7, 7) = 1e-3f;

    cv::Mat measurementNoiseCov = cv::Mat::zeros(MP, MP, Mat_t(1));
    measurementNoiseCov.at<track_t>(0, 0) = 1e-3f;
    measurementNoiseCov.at<track_t>(1, 1) = 1e-3f;
    measurementNoiseCov.at<track_t>(2, 2) = 1e-3f;
    measurementNoiseCov.at<track_t>(3, 3) = 1e-3f;

    cv::Mat initState(DP, 1, Mat_t(1));
    initState.at<track_t>(0, 0) = rect0.x;
    initState.at<track_t>(1, 0) = rect0.y;
    initState.at<track_t>(2, 0) = rectv0.x;
    initState.at<track_t>(3, 0) = rectv0.y;
    initState.at<track_t>(4, 0) = 0;
    initState.at<track_t>(5, 0) = 0;
    initState.at<track_t>(6, 0) = rect0.width;
    initState.at<track_t>(7, 0) = rect0.height;

    cv::Mat P = 1e-3f * cv::Mat::eye(DP, DP, Mat_t(1));

    cv::Ptr<AcceleratedModel> model(new AcceleratedModel(m_deltaTime, UkfModelKind::Rect));
    kalman::UnscentedKalmanFilterParams params(DP, MP, CP, 0, 0, model);
    params.dataType = Mat_t(1);
    params.stateInit = initState.clone();
    params.errorCovInit = P.clone();
    params.measurementNoiseCov = measurementNoiseCov.clone();
    params.processNoiseCov = processNoiseCov.clone();

    params.alpha = 1;
    params.beta = 2.0;
    params.k = -2.0;

    m_uncsentedKalman = kalman::createUnscentedKalmanFilter(params);

    m_initialized = true;
}

//---------------------------------------------------------------------------
void TKalmanFilter::CreateAugmentedUnscented(Point_t xy0, Point_t xyv0)
{
    // State layout: x, y, vx, vy, ax, ay
    m_velocityXInd = 2; // vx
    m_velocityYInd = 3; // vy
    int MP = 2;
    int DP = 6;
    int CP = 0;

    cv::Mat processNoiseCov = cv::Mat::zeros(DP, DP, Mat_t(1));
    for (int i = 0; i < DP; ++i)
        processNoiseCov.at<track_t>(i, i) = 1e-3f;

    cv::Mat measurementNoiseCov = cv::Mat::zeros(MP, MP, Mat_t(1));
    measurementNoiseCov.at<track_t>(0, 0) = 1e-6f;
    measurementNoiseCov.at<track_t>(1, 1) = 1e-6f;

    cv::Mat initState(DP, 1, Mat_t(1));
    initState.at<track_t>(0, 0) = xy0.x;
    initState.at<track_t>(1, 0) = xy0.y;
    initState.at<track_t>(2, 0) = xyv0.x;
    initState.at<track_t>(3, 0) = xyv0.y;
    initState.at<track_t>(4, 0) = 0;
    initState.at<track_t>(5, 0) = 0;

    cv::Mat P = 1e-6f * cv::Mat::eye(DP, DP, Mat_t(1));

    cv::Ptr<AcceleratedModel> model(new AcceleratedModel(m_deltaTime, UkfModelKind::Point));
    kalman::AugmentedUnscentedKalmanFilterParams params(DP, MP, CP, 0, 0, model);
    params.dataType = Mat_t(1);
    params.stateInit = initState.clone();
    params.errorCovInit = P.clone();
    params.measurementNoiseCov = measurementNoiseCov.clone();
    params.processNoiseCov = processNoiseCov.clone();

    params.alpha = 1;
    params.beta = 2.0;
    params.k = -2.0;

    m_uncsentedKalman = kalman::createAugmentedUnscentedKalmanFilter(params);

    m_initialized = true;
}

//---------------------------------------------------------------------------
void TKalmanFilter::CreateAugmentedUnscented(cv::Rect_<track_t> rect0, Point_t rectv0)
{
    // State layout: x, y, vx, vy, ax, ay, width, height
    m_velocityXInd = 2; // vx
    m_velocityYInd = 3; // vy
    int MP = 4;
    int DP = 8;
    int CP = 0;

    cv::Mat processNoiseCov = cv::Mat::zeros(DP, DP, Mat_t(1));
    processNoiseCov.at<track_t>(0, 0) = 1e-3f;
    processNoiseCov.at<track_t>(1, 1) = 1e-3f;
    processNoiseCov.at<track_t>(2, 2) = 1e-3f;
    processNoiseCov.at<track_t>(3, 3) = 1e-3f;
    processNoiseCov.at<track_t>(4, 4) = 1e-3f;
    processNoiseCov.at<track_t>(5, 5) = 1e-3f;
    processNoiseCov.at<track_t>(6, 6) = 1e-3f;
    processNoiseCov.at<track_t>(7, 7) = 1e-3f;

    cv::Mat measurementNoiseCov = cv::Mat::zeros(MP, MP, Mat_t(1));
    measurementNoiseCov.at<track_t>(0, 0) = 1e-3f;
    measurementNoiseCov.at<track_t>(1, 1) = 1e-3f;
    measurementNoiseCov.at<track_t>(2, 2) = 1e-3f;
    measurementNoiseCov.at<track_t>(3, 3) = 1e-3f;

    cv::Mat initState(DP, 1, Mat_t(1));
    initState.at<track_t>(0, 0) = rect0.x;
    initState.at<track_t>(1, 0) = rect0.y;
    initState.at<track_t>(2, 0) = rectv0.x;
    initState.at<track_t>(3, 0) = rectv0.y;
    initState.at<track_t>(4, 0) = 0;
    initState.at<track_t>(5, 0) = 0;
    initState.at<track_t>(6, 0) = rect0.width;
    initState.at<track_t>(7, 0) = rect0.height;

    cv::Mat P = 1e-3f * cv::Mat::eye(DP, DP, Mat_t(1));

    cv::Ptr<AcceleratedModel> model(new AcceleratedModel(m_deltaTime, UkfModelKind::Rect));
    kalman::AugmentedUnscentedKalmanFilterParams params(DP, MP, CP, 0, 0, model);
    params.dataType = Mat_t(1);
    params.stateInit = initState.clone();
    params.errorCovInit = P.clone();
    params.measurementNoiseCov = measurementNoiseCov.clone();
    params.processNoiseCov = processNoiseCov.clone();

    params.alpha = 1;
    params.beta = 2.0;
    params.k = -2.0;

    m_uncsentedKalman = kalman::createAugmentedUnscentedKalmanFilter(params);

    m_initialized = true;
}

//---------------------------------------------------------------------------
void TKalmanFilter::CreateUnscented(cv::RotatedRect rrect0, Point_t rrectv0)
{
    // State layout: x, y, vx, vy, ax, ay, width, height, angle, vangle
    m_velocityXInd = 2; // vx
    m_velocityYInd = 3; // vy
    int MP = 5;
    int DP = 10;
    int CP = 0;

    cv::Mat processNoiseCov = cv::Mat::zeros(DP, DP, Mat_t(1));
    for (int i = 0; i < DP; ++i)
        processNoiseCov.at<track_t>(i, i) = 1e-3f;

    cv::Mat measurementNoiseCov = cv::Mat::zeros(MP, MP, Mat_t(1));
    for (int i = 0; i < MP; ++i)
        measurementNoiseCov.at<track_t>(i, i) = 1e-3f;

    cv::Mat initState(DP, 1, Mat_t(1));
    initState.at<track_t>(0, 0) = rrect0.center.x;
    initState.at<track_t>(1, 0) = rrect0.center.y;
    initState.at<track_t>(2, 0) = rrectv0.x;
    initState.at<track_t>(3, 0) = rrectv0.y;
    initState.at<track_t>(4, 0) = 0;
    initState.at<track_t>(5, 0) = 0;
    initState.at<track_t>(6, 0) = rrect0.size.width;
    initState.at<track_t>(7, 0) = rrect0.size.height;
    initState.at<track_t>(8, 0) = rrect0.angle;
    initState.at<track_t>(9, 0) = 0;

    cv::Mat P = 1e-3f * cv::Mat::eye(DP, DP, Mat_t(1));

    cv::Ptr<AcceleratedModel> model(new AcceleratedModel(m_deltaTime, UkfModelKind::RRect));
    kalman::UnscentedKalmanFilterParams params(DP, MP, CP, 0, 0, model);
    params.dataType = Mat_t(1);
    params.stateInit = initState.clone();
    params.errorCovInit = P.clone();
    params.measurementNoiseCov = measurementNoiseCov.clone();
    params.processNoiseCov = processNoiseCov.clone();

    params.alpha = 1;
    params.beta = 2.0;
    params.k = -2.0;

    m_uncsentedKalman = kalman::createUnscentedKalmanFilter(params);

    m_initialized = true;
}

//---------------------------------------------------------------------------
void TKalmanFilter::CreateAugmentedUnscented(cv::RotatedRect rrect0, Point_t rrectv0)
{
    // State layout: x, y, vx, vy, ax, ay, width, height, angle, vangle
    m_velocityXInd = 2; // vx
    m_velocityYInd = 3; // vy
    int MP = 5;
    int DP = 10;
    int CP = 0;

    cv::Mat processNoiseCov = cv::Mat::zeros(DP, DP, Mat_t(1));
    for (int i = 0; i < DP; ++i)
        processNoiseCov.at<track_t>(i, i) = 1e-3f;

    cv::Mat measurementNoiseCov = cv::Mat::zeros(MP, MP, Mat_t(1));
    for (int i = 0; i < MP; ++i)
        measurementNoiseCov.at<track_t>(i, i) = 1e-3f;

    cv::Mat initState(DP, 1, Mat_t(1));
    initState.at<track_t>(0, 0) = rrect0.center.x;
    initState.at<track_t>(1, 0) = rrect0.center.y;
    initState.at<track_t>(2, 0) = rrectv0.x;
    initState.at<track_t>(3, 0) = rrectv0.y;
    initState.at<track_t>(4, 0) = 0;
    initState.at<track_t>(5, 0) = 0;
    initState.at<track_t>(6, 0) = rrect0.size.width;
    initState.at<track_t>(7, 0) = rrect0.size.height;
    initState.at<track_t>(8, 0) = rrect0.angle;
    initState.at<track_t>(9, 0) = 0;

    cv::Mat P = 1e-3f * cv::Mat::eye(DP, DP, Mat_t(1));

    cv::Ptr<AcceleratedModel> model(new AcceleratedModel(m_deltaTime, UkfModelKind::RRect));
    kalman::AugmentedUnscentedKalmanFilterParams params(DP, MP, CP, 0, 0, model);
    params.dataType = Mat_t(1);
    params.stateInit = initState.clone();
    params.errorCovInit = P.clone();
    params.measurementNoiseCov = measurementNoiseCov.clone();
    params.processNoiseCov = processNoiseCov.clone();

    params.alpha = 1;
    params.beta = 2.0;
    params.k = -2.0;

    m_uncsentedKalman = kalman::createAugmentedUnscentedKalmanFilter(params);

    m_initialized = true;
}

#endif

///
/// \brief TKalmanFilter::GetPointPrediction
/// \return
///
Point_t TKalmanFilter::GetPointPrediction()
{
    if (m_initialized)
    {
        cv::Mat prediction;

        switch (m_type)
        {
        case tracking::KalmanLinear:
            prediction = m_linearKalman.predict();
            break;

        case tracking::KalmanUnscented:
        case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
            prediction = m_uncsentedKalman->predict();
#else
            prediction = m_linearKalman.predict();
            std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
            break;
        }

        m_lastPointResult = Point_t(prediction.at<track_t>(0), prediction.at<track_t>(1));
    }
    return m_lastPointResult;
}

///
/// \brief TKalmanFilter::Update
/// \param pt
/// \param dataCorrect
/// \return
///
Point_t TKalmanFilter::Update(Point_t pt, bool dataCorrect)
{
    if (!m_initialized)
    {
        if (m_initialPoints.size() < MIN_INIT_VALS)
        {
            if (dataCorrect)
            {
                m_initialPoints.push_back(pt);
                m_lastPointResult = pt;
            }
        }
        if (m_initialPoints.size() >= MIN_INIT_VALS)
        {
            // Two samples are enough to bootstrap the model: start from the
            // latest position with the finite-difference velocity.
            const Point_t xy0 = m_initialPoints.back();
            const Point_t xyv0 = m_initialPoints.back() - m_initialPoints.front();

            switch (m_type)
            {
            case tracking::KalmanLinear:
                if (m_useAcceleration)
					CreateLinearAcceleration(xy0, xyv0);
				else
					CreateLinear(xy0, xyv0);
                break;

            case tracking::KalmanUnscented:
#ifdef USE_OCV_UKF
                CreateUnscented(xy0, xyv0);
#else
				if (m_useAcceleration)
					CreateLinearAcceleration(xy0, xyv0);
				else
					CreateLinear(xy0, xyv0);
                std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
                break;

            case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
                CreateAugmentedUnscented(xy0, xyv0);
#else
				if (m_useAcceleration)
					CreateLinearAcceleration(xy0, xyv0);
				else
					CreateLinear(xy0, xyv0);
                std::cerr << "AugmentedUnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
                break;
            }
            m_lastDist = 0;
        }
    }

    if (m_initialized)
    {
        cv::Mat measurement(2, 1, Mat_t(1));
        if (!dataCorrect)
        {
            measurement.at<track_t>(0) = m_lastPointResult.x;  //update using prediction
            measurement.at<track_t>(1) = m_lastPointResult.y;
        }
        else
        {
            measurement.at<track_t>(0) = pt.x;  //update using measurements
            measurement.at<track_t>(1) = pt.y;
        }
        // Correction
        cv::Mat estimated;
        switch (m_type)
        {
        case tracking::KalmanLinear:
        {
            estimated = m_linearKalman.correct(measurement);

            // Inertia correction. Only adapt when a real measurement is
			// available: on lost frames pt is an empty region (0, 0) and the
			// distance to it would push m_deltaTime to its maximum, distorting
			// the transition matrix precisely while the track is lost.
			if (dataCorrect && !m_useAcceleration)
			{
				track_t currDist = sqrtf(sqr(estimated.at<track_t>(0) - pt.x) + sqr(estimated.at<track_t>(1) - pt.y));
				if (currDist > m_lastDist)
					m_deltaTime = std::min(m_deltaTime + m_deltaStep, m_deltaTimeMax);
				else
					m_deltaTime = std::max(m_deltaTime - m_deltaStep, m_deltaTimeMin);

				m_lastDist = currDist;

				m_linearKalman.transitionMatrix.at<track_t>(0, 2) = m_deltaTime;
				m_linearKalman.transitionMatrix.at<track_t>(1, 3) = m_deltaTime;
			}
            break;
        }

        case tracking::KalmanUnscented:
        case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
            estimated = m_uncsentedKalman->correct(measurement);
#else
            estimated = m_linearKalman.correct(measurement);
            std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
            break;
        }

        m_lastPointResult.x = estimated.at<track_t>(0);   //update using measurements
        m_lastPointResult.y = estimated.at<track_t>(1);
    }
    else
    {
        if (dataCorrect)
            m_lastPointResult = pt;
    }
    return m_lastPointResult;
}

///
/// \brief TKalmanFilter::GetRectPrediction
/// \return
///
cv::Rect TKalmanFilter::GetRectPrediction()
{
    if (m_initialized)
    {
        cv::Mat prediction;

        switch (m_type)
        {
        case tracking::KalmanLinear:
            prediction = m_linearKalman.predict();
            m_lastRectResult = cv::Rect_<track_t>(prediction.at<track_t>(0), prediction.at<track_t>(1), prediction.at<track_t>(2), prediction.at<track_t>(3));
            break;

        case tracking::KalmanUnscented:
        case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
            prediction = m_uncsentedKalman->predict();
            // UKF rect state is [x, y, vx, vy, ax, ay, width, height],
            // so width/height are at indices 6 and 7, not 2 and 3.
            m_lastRectResult = cv::Rect_<track_t>(prediction.at<track_t>(0), prediction.at<track_t>(1), prediction.at<track_t>(6), prediction.at<track_t>(7));
#else
            prediction = m_linearKalman.predict();
            m_lastRectResult = cv::Rect_<track_t>(prediction.at<track_t>(0), prediction.at<track_t>(1), prediction.at<track_t>(2), prediction.at<track_t>(3));
            std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
            break;
        }
    }
    return cv::Rect(static_cast<int>(m_lastRectResult.x), static_cast<int>(m_lastRectResult.y), static_cast<int>(m_lastRectResult.width), static_cast<int>(m_lastRectResult.height));
}

///
/// \brief TKalmanFilter::Update
/// \param rect
/// \param dataCorrect
/// \return
///
cv::Rect TKalmanFilter::Update(cv::Rect rect, bool dataCorrect)
{
    if (!m_initialized)
    {
        if (m_initialRects.size() < MIN_INIT_VALS)
        {
            if (dataCorrect)
            {
                m_initialRects.push_back(rect);
                m_lastRectResult.x = static_cast<track_t>(rect.x);
                m_lastRectResult.y = static_cast<track_t>(rect.y);
                m_lastRectResult.width = static_cast<track_t>(rect.width);
                m_lastRectResult.height = static_cast<track_t>(rect.height);
            }
        }
        if (m_initialRects.size() >= MIN_INIT_VALS)
        {
            Point_t averageSize(0, 0);
            for (const auto& r : m_initialRects)
            {
                averageSize.x += static_cast<track_t>(r.width);
                averageSize.y += static_cast<track_t>(r.height);
            }
            averageSize.x /= MIN_INIT_VALS;
            averageSize.y /= MIN_INIT_VALS;

            // Two samples are enough to bootstrap the model: start from the
            // latest position with the finite-difference velocity.
            const auto& first = m_initialRects.front();
            const auto& last = m_initialRects.back();
            cv::Rect_<track_t> rect0(static_cast<track_t>(last.x), static_cast<track_t>(last.y), averageSize.x, averageSize.y);
            Point_t rectv0(static_cast<track_t>(last.x - first.x), static_cast<track_t>(last.y - first.y));

            switch (m_type)
            {
            case tracking::KalmanLinear:
				if (m_useAcceleration)
					CreateLinearAcceleration(rect0, rectv0);
				else
					CreateLinear(rect0, rectv0);
                break;

            case tracking::KalmanUnscented:
#ifdef USE_OCV_UKF
                CreateUnscented(rect0, rectv0);
#else
				if (m_useAcceleration)
					CreateLinearAcceleration(rect0, rectv0);
				else
					CreateLinear(rect0, rectv0);
                std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
                break;

            case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
                CreateAugmentedUnscented(rect0, rectv0);
#else
				if (m_useAcceleration)
					CreateLinearAcceleration(rect0, rectv0);
				else
					CreateLinear(rect0, rectv0);
                std::cerr << "AugmentedUnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
                break;
            }
        }
    }

    if (m_initialized)
    {
        cv::Mat measurement(4, 1, Mat_t(1));
        if (!dataCorrect)
        {
            measurement.at<track_t>(0) = m_lastRectResult.x;  // update using prediction
            measurement.at<track_t>(1) = m_lastRectResult.y;
            measurement.at<track_t>(2) = m_lastRectResult.width;
            measurement.at<track_t>(3) = m_lastRectResult.height;
        }
        else
        {
            measurement.at<track_t>(0) = static_cast<track_t>(rect.x);  // update using measurements
            measurement.at<track_t>(1) = static_cast<track_t>(rect.y);
            measurement.at<track_t>(2) = static_cast<track_t>(rect.width);
            measurement.at<track_t>(3) = static_cast<track_t>(rect.height);
        }
        // Correction
        cv::Mat estimated;
        switch (m_type)
        {
        case tracking::KalmanLinear:
        {
            estimated = m_linearKalman.correct(measurement);

            m_lastRectResult.x = estimated.at<track_t>(0);   //update using measurements
            m_lastRectResult.y = estimated.at<track_t>(1);
            m_lastRectResult.width = estimated.at<track_t>(2);
            m_lastRectResult.height = estimated.at<track_t>(3);

            // Inertia correction. Skip lost frames: rect is then an empty
			// region (0, 0, 0, 0) and the distance to it has no meaning.
			if (dataCorrect && !m_useAcceleration)
			{
				track_t currDist = sqrtf(sqr(estimated.at<track_t>(0) - rect.x) + sqr(estimated.at<track_t>(1) - rect.y) + sqr(estimated.at<track_t>(2) - rect.width) + sqr(estimated.at<track_t>(3) - rect.height));
				if (currDist > m_lastDist)
					m_deltaTime = std::min(m_deltaTime + m_deltaStep, m_deltaTimeMax);
				else
					m_deltaTime = std::max(m_deltaTime - m_deltaStep, m_deltaTimeMin);

				m_lastDist = currDist;

				m_linearKalman.transitionMatrix.at<track_t>(0, 4) = m_deltaTime;
				m_linearKalman.transitionMatrix.at<track_t>(1, 5) = m_deltaTime;
				m_linearKalman.transitionMatrix.at<track_t>(2, 6) = m_deltaTime;
				m_linearKalman.transitionMatrix.at<track_t>(3, 7) = m_deltaTime;
			}
            break;
        }

        case tracking::KalmanUnscented:
        case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
            estimated = m_uncsentedKalman->correct(measurement);

            m_lastRectResult.x = estimated.at<track_t>(0);   //update using measurements
            m_lastRectResult.y = estimated.at<track_t>(1);
            m_lastRectResult.width = estimated.at<track_t>(6);
            m_lastRectResult.height = estimated.at<track_t>(7);
#else
            estimated = m_linearKalman.correct(measurement);

            m_lastRectResult.x = estimated.at<track_t>(0);   //update using measurements
            m_lastRectResult.y = estimated.at<track_t>(1);
            m_lastRectResult.width = estimated.at<track_t>(2);
            m_lastRectResult.height = estimated.at<track_t>(3);
            std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
            break;
        }
    }
    else
    {
        if (dataCorrect)
        {
            m_lastRectResult.x = static_cast<track_t>(rect.x);
            m_lastRectResult.y = static_cast<track_t>(rect.y);
            m_lastRectResult.width = static_cast<track_t>(rect.width);
            m_lastRectResult.height = static_cast<track_t>(rect.height);
        }
    }
    return cv::Rect(static_cast<int>(m_lastRectResult.x), static_cast<int>(m_lastRectResult.y), static_cast<int>(m_lastRectResult.width), static_cast<int>(m_lastRectResult.height));
}

///
/// \brief TKalmanFilter::GetRRectPrediction
/// \return
///
cv::RotatedRect TKalmanFilter::GetRRectPrediction()
{
	if (m_initialized)
	{
		cv::Mat prediction;

		switch (m_type)
		{
		case tracking::KalmanLinear:
			prediction = m_linearKalman.predict();
			m_lastRRectResult.center.x = prediction.at<track_t>(0);
			m_lastRRectResult.center.y = prediction.at<track_t>(1);
			m_lastRRectResult.size.width = prediction.at<track_t>(2);
			m_lastRRectResult.size.height = prediction.at<track_t>(3);
			m_lastRRectResult.angle = prediction.at<track_t>(4);
			break;

		case tracking::KalmanUnscented:
		case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
			prediction = m_uncsentedKalman->predict();
			// UKF rrect state: [x, y, vx, vy, ax, ay, width, height, angle, vangle]
			m_lastRRectResult.center.x = prediction.at<track_t>(0);
			m_lastRRectResult.center.y = prediction.at<track_t>(1);
			m_lastRRectResult.size.width = prediction.at<track_t>(6);
			m_lastRRectResult.size.height = prediction.at<track_t>(7);
			m_lastRRectResult.angle = prediction.at<track_t>(8);
#else
			prediction = m_linearKalman.predict();
			m_lastRRectResult.center.x = prediction.at<track_t>(0);
			m_lastRRectResult.center.y = prediction.at<track_t>(1);
			m_lastRRectResult.size.width = prediction.at<track_t>(2);
			m_lastRRectResult.size.height = prediction.at<track_t>(3);
			m_lastRRectResult.angle = prediction.at<track_t>(4);
			std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
			break;
		}
	}
	return NormalizeRotatedRectAngle(m_lastRRectResult);
}

///
/// \brief TKalmanFilter::Update
/// \param rrect
/// \param dataCorrect
/// \return
///
cv::RotatedRect TKalmanFilter::Update(cv::RotatedRect rrect, bool dataCorrect)
{
	if (!m_initialized)
	{
		if (m_initialRRects.size() < MIN_INIT_VALS)
		{
			if (dataCorrect)
			{
				m_initialRRects.push_back(rrect);
				m_lastRRectResult = rrect;
			}
		}
		if (m_initialRRects.size() >= MIN_INIT_VALS)
		{
			Point_t averageSize(0, 0);
			for (const auto& rr : m_initialRRects)
			{
				averageSize.x += rr.size.width;
				averageSize.y += rr.size.height;
			}
			averageSize.x /= MIN_INIT_VALS;
			averageSize.y /= MIN_INIT_VALS;
			// Orientations are undirected (period 180 degrees), so the arithmetic
			// mean is meaningless near the +-90 degree boundary.
			const track_t averageAngle = CircularMeanHalfTurnDeg(m_initialRRects);

			const cv::RotatedRect& first = m_initialRRects.front();
			const cv::RotatedRect& last = m_initialRRects.back();
			cv::RotatedRect rrect0(cv::Point2f(last.center.x, last.center.y), averageSize, averageAngle);
			Point_t rrectv0(last.center.x - first.center.x, last.center.y - first.center.y);

			switch (m_type)
			{
			case tracking::KalmanLinear:
				if (m_useAcceleration)
					CreateLinearAcceleration(rrect0, rrectv0);
				else
					CreateLinear(rrect0, rrectv0);
				break;

			case tracking::KalmanUnscented:
#ifdef USE_OCV_UKF
				CreateUnscented(rrect0, rrectv0);
#else
				if (m_useAcceleration)
					CreateLinearAcceleration(rrect0, rrectv0);
				else
					CreateLinear(rrect0, rrectv0);
				std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
				break;

			case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
				CreateAugmentedUnscented(rrect0, rrectv0);
#else
				if (m_useAcceleration)
					CreateLinearAcceleration(rrect0, rrectv0);
				else
					CreateLinear(rrect0, rrectv0);
				std::cerr << "AugmentedUnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
				break;
			}
		}
	}

	if (m_initialized)
	{
		cv::Mat measurement(5, 1, Mat_t(1));
		if (!dataCorrect)
		{
			measurement.at<track_t>(0) = m_lastRRectResult.center.x;  // update using prediction
			measurement.at<track_t>(1) = m_lastRRectResult.center.y;
			measurement.at<track_t>(2) = m_lastRRectResult.size.width;
			measurement.at<track_t>(3) = m_lastRRectResult.size.height;
			measurement.at<track_t>(4) = m_lastRRectResult.angle;
		}
		else
		{
			measurement.at<track_t>(0) = static_cast<track_t>(rrect.center.x);  // update using measurements
			measurement.at<track_t>(1) = static_cast<track_t>(rrect.center.y);
			measurement.at<track_t>(2) = static_cast<track_t>(rrect.size.width);
			measurement.at<track_t>(3) = static_cast<track_t>(rrect.size.height);
			// RotatedRect orientation is undirected and the filters keep the
			// angle continuous, so unroll the measured angle by 180 degree steps
			// towards the prediction. Otherwise crossing the +-90 degree boundary
			// looks like a huge rotation to the filter.
			track_t measuredAngle = static_cast<track_t>(rrect.angle);
			if (m_linearKalman.statePre.rows > 4)
			{
				// Linear rrect state: [x, y, width, height, angle, ...]
				measuredAngle = UnwrapHalfTurnDeg(measuredAngle, m_linearKalman.statePre.at<track_t>(4));
			}
#ifdef USE_OCV_UKF
			else if (m_uncsentedKalman)
			{
				// UKF rrect state: [x, y, vx, vy, ax, ay, width, height, angle, vangle]
				const cv::Mat ukfState = m_uncsentedKalman->getState();
				if (ukfState.rows > 8)
					measuredAngle = UnwrapHalfTurnDeg(measuredAngle, ukfState.at<track_t>(8));
			}
#endif
			measurement.at<track_t>(4) = measuredAngle;
		}
		// Correction
		cv::Mat estimated;
		switch (m_type)
		{
		case tracking::KalmanLinear:
		{
			estimated = m_linearKalman.correct(measurement);

			m_lastRRectResult.center.x = estimated.at<track_t>(0);   //update using measurements
			m_lastRRectResult.center.y = estimated.at<track_t>(1);
			m_lastRRectResult.size.width = estimated.at<track_t>(2);
			m_lastRRectResult.size.height = estimated.at<track_t>(3);
			m_lastRRectResult.angle = estimated.at<track_t>(4);

			// Inertia correction. Skip lost frames: rrect is then an empty
			// region (0, 0, 0, 0) and the distance to it has no meaning.
			if (dataCorrect && !m_useAcceleration)
			{
				track_t currDist = sqrtf(sqr(estimated.at<track_t>(0) - rrect.center.x) + sqr(estimated.at<track_t>(1) - rrect.center.y) +
					               sqr(estimated.at<track_t>(2) - rrect.size.width) + sqr(estimated.at<track_t>(3) - rrect.size.height));
				if (currDist > m_lastDist)
					m_deltaTime = std::min(m_deltaTime + m_deltaStep, m_deltaTimeMax);
				else
					m_deltaTime = std::max(m_deltaTime - m_deltaStep, m_deltaTimeMin);

				m_lastDist = currDist;

				m_linearKalman.transitionMatrix.at<track_t>(0, 5) = m_deltaTime;
				m_linearKalman.transitionMatrix.at<track_t>(1, 6) = m_deltaTime;
				m_linearKalman.transitionMatrix.at<track_t>(2, 7) = m_deltaTime;
				m_linearKalman.transitionMatrix.at<track_t>(3, 8) = m_deltaTime;
				m_linearKalman.transitionMatrix.at<track_t>(4, 9) = m_deltaTime;
			}
			break;
		}

		case tracking::KalmanUnscented:
		case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
			estimated = m_uncsentedKalman->correct(measurement);

			m_lastRRectResult.center.x = estimated.at<track_t>(0);   //update using measurements
			m_lastRRectResult.center.y = estimated.at<track_t>(1);
			m_lastRRectResult.size.width = estimated.at<track_t>(6);
			m_lastRRectResult.size.height = estimated.at<track_t>(7);
			m_lastRRectResult.angle = estimated.at<track_t>(8);
#else
			estimated = m_linearKalman.correct(measurement);

			m_lastRRectResult.center.x = estimated.at<track_t>(0);   //update using measurements
			m_lastRRectResult.center.y = estimated.at<track_t>(1);
			m_lastRRectResult.size.width = estimated.at<track_t>(2);
			m_lastRRectResult.size.height = estimated.at<track_t>(3);
			m_lastRRectResult.angle = estimated.at<track_t>(4);
			std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
			break;
		}
	}
	else
	{
		if (dataCorrect)
			m_lastRRectResult = rrect;
	}
	return NormalizeRotatedRectAngle(m_lastRRectResult);
}

///
/// \brief TKalmanFilter::GetVelocity
/// \return
///
cv::Vec<track_t, 2> TKalmanFilter::GetVelocity() const
{
    cv::Vec<track_t, 2> res(0, 0);
    if (m_initialized)
    {
        switch (m_type)
        {
        case tracking::KalmanLinear:
        {
            // The state layout depends on the created model (point/rect/rrect,
            // with or without acceleration), so use the recorded indices.
            if (m_linearKalman.statePre.rows > m_velocityYInd)
            {
                res[0] = m_linearKalman.statePre.at<track_t>(m_velocityXInd);
                res[1] = m_linearKalman.statePre.at<track_t>(m_velocityYInd);
            }
            break;
        }

        case tracking::KalmanUnscented:
        case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
            cv::Mat state = m_uncsentedKalman->getState();
            if (state.rows > m_velocityYInd)
            {
                res[0] = state.at<track_t>(m_velocityXInd);
                res[1] = state.at<track_t>(m_velocityYInd);
            }
#else
            std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
            break;
        }
    }
    return res;
}

//---------------------------------------------------------------------------
void TKalmanFilter::GetPredictionAndInnovCov(cv::Mat& innovCov, cv::Mat& predictedMeas) const
{
    innovCov.release();
    predictedMeas.release();

    if (!m_initialized)
        return;

    switch (m_type)
    {
    case tracking::KalmanLinear:
    {
        // statePre and errorCovPre are filled by predict() right before the
        // assignment matrix is built. Use the measurement matrix (identity
        // MP x DP for every model) to map the state to the measurement space:
        //   z = H * x,  S = H * P * H^T + R
        const cv::Mat& H = m_linearKalman.measurementMatrix;
        if (H.empty() || m_linearKalman.statePre.empty() || m_linearKalman.errorCovPre.empty())
            break;

        predictedMeas = H * m_linearKalman.statePre;
        innovCov = H * m_linearKalman.errorCovPre * H.t() + m_linearKalman.measurementNoiseCov;
        break;
    }

    case tracking::KalmanUnscented:
    case tracking::KalmanAugmentedUnscented:
    {
#ifdef USE_OCV_UKF
        if (!m_uncsentedKalman)
            break;

        const cv::Mat state = m_uncsentedKalman->getState();
        const cv::Mat errorCov = m_uncsentedKalman->getErrorCov();
        const cv::Mat measurementNoiseCov = m_uncsentedKalman->getMeasurementNoiseCov();
        if (state.empty() || errorCov.empty() || measurementNoiseCov.empty())
            break;

        const int dp = state.rows;
        const int mp = measurementNoiseCov.rows;

        // The UKF measurement function is the identity on the measured
        // components, so H is a selection matrix. The state layouts are:
        //   point: x, y, vx, vy, ax, ay                       (mp = 2)
        //   rect:  x, y, vx, vy, ax, ay, w, h                 (mp = 4)
        //   rrect: x, y, vx, vy, ax, ay, w, h, angle, vangle  (mp = 5)
        cv::Mat H = cv::Mat::zeros(mp, dp, Mat_t(1));
        if (mp == 2 && dp == 6)
        {
            H.at<track_t>(0, 0) = 1;
            H.at<track_t>(1, 1) = 1;
        }
        else if (mp == 4 && dp == 8)
        {
            H.at<track_t>(0, 0) = 1;
            H.at<track_t>(1, 1) = 1;
            H.at<track_t>(2, 6) = 1;
            H.at<track_t>(3, 7) = 1;
        }
        else if (mp == 5 && dp == 10)
        {
            H.at<track_t>(0, 0) = 1;
            H.at<track_t>(1, 1) = 1;
            H.at<track_t>(2, 6) = 1;
            H.at<track_t>(3, 7) = 1;
            H.at<track_t>(4, 8) = 1;
        }
        else
        {
            // Unexpected state/measurement layout: do not invent a metric.
            break;
        }

        predictedMeas = H * state;
        innovCov = H * errorCov * H.t() + measurementNoiseCov;
#endif
        break;
    }
    }
}

