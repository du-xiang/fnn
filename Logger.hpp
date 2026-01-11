#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <fstream>

enum class logLevel
{
    logINFO,
    logWARN,
    logERROR,
    logFATAL
};

class Logger
{
private:
    logLevel m_logLevel;
    std::ofstream m_logFile;

    Logger() : m_logLevel(logLevel::logINFO) {};       // 默认logLevel为INFO
    ~Logger();
public:
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    static Logger& getInstance(const std::string& filePath);
    bool set_log_level(logLevel level);
    bool set_log_file(const std::string& filePath);
    bool log(logLevel level, const std::string& file, int line, const std::string& message);
};

#endif // !LOGGER_H