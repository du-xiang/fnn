#include <iostream>
#include <cmath>
#include <cstdio>

#include "Util.hpp"

double sigmoid(const double& x) {
	return 1.0 / (1.0 + exp(-x));
}

// **********************************************
// 进度条类成员函数实现
ProgressBar::ProgressBar(std::size_t total, 
                std::size_t barWidth = 50, 
                std::string prefix   = "", 
                std::string suffix	 = "")
		: m_total(total),
		  m_bar_width(barWidth),
		  m_prefix(prefix),
		  m_suffix(suffix),
		  m_current(0)
{ }

ProgressBar::~ProgressBar()
{
	if(m_current == m_total)
		putchar('\n');
}

void ProgressBar::update(std::size_t now)
{
	m_current = now;
	double ratio = static_cast<double>(now) / m_total;
	std::size_t pos = static_cast<std::size_t>(ratio * m_bar_width);

	printf("\r%s[", m_prefix.c_str());
    for (std::size_t i = 0; i < m_bar_width; ++i)
        putchar(i < pos ? '=' : ' ');
    printf("] %zu/%zu %s", now, m_total, m_suffix.c_str());
    fflush(stdout);
}


// **********************************************
// 计时器类成员函数实现
Timer::Timer() noexcept
{ reset();}

void Timer::reset() noexcept
{
	m_start = std::chrono::steady_clock::now();
    m_paused = false;
    m_paused_sum = std::chrono::milliseconds{0};
}

void Timer::pause() noexcept
{
	if(!m_paused)
    {
        m_paused_sum += std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - m_start);
        m_paused = true;
    }
}

void Timer::resume() noexcept
{
    if(m_paused)
    {
        m_start = std::chrono::steady_clock::now();
        m_paused = false;
    }
}

long long Timer::elapsedTime() const noexcept
{
	auto extra = m_paused ? 
		std::chrono::milliseconds{0}: 
			std::chrono::duration_cast<std::chrono::milliseconds>(
                            		std::chrono::steady_clock::now() - m_start);

    return (m_paused_sum + extra).count();
}

bool Timer::isPause() const noexcept
{ return m_paused;}
