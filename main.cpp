#include<iostream>

#include "FullConnNN.hpp"
#include "Util.hpp"

int main()
{
	FullConnNN* example = new FullConnNN();
	// example->weight_init();

	Timer t;
	example->backward();
	t.pause();
	std::cout << "time: " + t.elapsedTime() << "ms" << std::endl;

	// example->display();
	return 0;
}