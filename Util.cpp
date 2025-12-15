#ifndef UTIL
#define UTIL

#include<iostream>
#include<cmath>

double sigmoid(const double& x) {
	return 1.0 / (1.0 + exp(-x));
}

#endif // !UTIL