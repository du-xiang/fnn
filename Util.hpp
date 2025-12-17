#ifndef UTIL_H
#define UTIL_H

#include <string>

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

#endif // !UTIL_H