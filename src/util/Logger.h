#pragma once

#include <chrono>
#include <ctime>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/rotating_file_sink.h"

#include "mtracking/defines.h"

namespace mtutil
{
///
/// \brief Create the console + rotating-file logger shared by the example apps.
///
/// Creates the log directory and tolerates a missing/unwritable one (the logger
/// then uses the console sink only). `level` is an spdlog level name, e.g.
/// "info" or "debug" (roadmap M15, M2).
///
inline std::shared_ptr<spdlog::logger> CreateLogger(const std::string& name,
                                                    const std::string& level,
                                                    const std::string& logDir = "logs")
{
    auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    consoleSink->set_level(spdlog::level::from_str(level));
    consoleSink->set_pattern("[%^%l%$] %v");

    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(consoleSink);

    std::error_code ec;
    fs::create_directories(logDir, ec);
    if (ec)
        std::cerr << "CreateLogger: can not create " << logDir << " directory: " << ec.message() << std::endl;

    const auto now = std::chrono::system_clock::now();
    const auto transformed = now.time_since_epoch().count() / 1000000;
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    char buffer[80];
#ifdef _WIN32
    tm timeInfo;
    localtime_s(&timeInfo, &tt);
    strftime(buffer, sizeof(buffer), "%G%m%d_%H%M%S", &timeInfo);
#else
    auto timeInfo = localtime(&tt);
    strftime(buffer, sizeof(buffer), "%G%m%d_%H%M%S", timeInfo);
#endif

    const size_t max_size = 1024 * 1024 * 5;
    const size_t max_files = 3;
    try
    {
        auto fileSink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            logDir + "/" + std::string(buffer) + std::to_string(transformed % 1000) + ".txt", max_size, max_files);
        fileSink->set_level(spdlog::level::from_str(level));
        sinks.push_back(fileSink);
    }
    catch (const std::exception& e)
    {
        std::cerr << "CreateLogger: can not create log file sink: " << e.what() << std::endl;
    }

    auto logger = std::make_shared<spdlog::logger>(name, sinks.begin(), sinks.end());
    logger->set_level(spdlog::level::from_str(level));
    logger->info("Start service");
    return logger;
}
} // namespace mtutil
