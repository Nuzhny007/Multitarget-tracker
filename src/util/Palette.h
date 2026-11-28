#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace mtutil
{
///
/// \brief Default track palette shared by the example apps.
///
inline std::vector<cv::Scalar> DefaultPalette()
{
    std::vector<cv::Scalar> colors;
    colors.reserve(9);
    colors.emplace_back(255, 0, 0);
    colors.emplace_back(0, 255, 0);
    colors.emplace_back(0, 0, 255);
    colors.emplace_back(255, 255, 0);
    colors.emplace_back(0, 255, 255);
    colors.emplace_back(255, 0, 255);
    colors.emplace_back(255, 127, 255);
    colors.emplace_back(127, 0, 255);
    colors.emplace_back(127, 0, 127);
    return colors;
}
} // namespace mtutil
