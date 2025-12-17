#include <iostream>
#include <cmath>
#include <cstdio>

#include "Util.hpp"

double sigmoid(const double& x) {
	return 1.0 / (1.0 + exp(-x));
}

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