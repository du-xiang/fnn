#ifndef UTIL_H
#define UTIL_H

#include <string>
#include <chrono>

class ProgressBar
{
private:
    std::size_t m_total;
    std::size_t m_bar_width;
    std::string m_prefix;
    std::string m_suffix;
    std::size_t m_current;
public:
    ProgressBar() = delete;
    ProgressBar(std::size_t total, 
                std::size_t barWidth, 
                std::string prefix, 
                std::string suffix);
    void update(std::size_t now);
    
};

class Timer
{
private:
    std::chrono::steady_clock::time_point m_start;
    std::chrono::milliseconds m_paused_sum;
    bool m_paused;
public:
    Timer() noexcept;
    void reset() noexcept;
    void pause() noexcept;
    void resume() noexcept;
    long long elapsedTime() const noexcept;
    bool isPause() const noexcept;
};

#endif // !UTIL_H