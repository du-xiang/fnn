#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>

#include "Logger.hpp"

Logger::~Logger()
{
    if(m_logFile.is_open())
    {
        m_logFile.close();
    }
}

Logger& Logger::getInstance(const std::string& filePath)
{
    static Logger instace;
    instace.set_log_file(filePath);
    return instace;
}

bool Logger::set_log_level(logLevel level)
{
    m_logLevel = level;
    return true;
}

bool Logger::set_log_file(const std::string& filePath)
{
    if(!m_logFile.is_open())
        m_logFile.open(filePath, std::ios::app);
    if(!m_logFile.is_open())
    {
        std::cerr << "[log error] log file open failed: " << filePath << std::endl;
        return false;
    }
    return true;
}

bool Logger::log(logLevel level, const std::string& file, int line, const std::string& message)
{
    auto now = std::chrono::system_clock::now();
    auto ms_total = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    std::time_t now_sec = ms_total / 1000;
    int ms = ms_total % 1000;
    std::ostringstream oss_time;
    oss_time << std::put_time(std::localtime(&now_sec), "%Y-%m-%d %H:%M:%S") 
             << "." << std::setfill('0') << std::setw(3) << ms;

    std::string logLevelStr = "";
    switch (level)
    {
        case logLevel::logINFO: logLevelStr = "INFO"; break;
        case logLevel::logWARN: logLevelStr = "WARN"; break;
        case logLevel::logERROR: logLevelStr = "ERROR"; break;
        case logLevel::logFATAL: logLevelStr = "FATAL"; break;
    }

    std::string logStr = "[" + oss_time.str() + "] "
                        + "[" + logLevelStr + "] "
                        + "[" + file + ":" + std::to_string(line) + "] - "
                        + message;

    if(m_logFile.is_open())
    {
        m_logFile << logStr << std::endl;
        m_logFile.flush();
    }

    return true;
}