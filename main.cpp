#include<iostream>

#include "FullConnNN.hpp"

int main()
{
	FullConnNN* example = new FullConnNN();
	// example->weight_init();

	example->backward();

	example->display();
	return 0;
}