#include<iostream>

#include "FullConnNN.hpp"

int main()
{
	FullConnNN* example = new FullConnNN();
	// example->weight_init();
	example->display();

	std::vector<double> in(2, 10);
	example->forward(in);
	// example->backward();

	return 0;
}