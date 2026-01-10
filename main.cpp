#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstdio>
#include <cstdint>

#include "FullConnNN.hpp"
#include "Util.hpp"
#include "Loader.hpp"

int main()
{
	std::string weightPath = "..//weight//weight.w";
	FullConnNN* example = new FullConnNN();

	if(example->weight_load(weightPath))
	{
		Sample sample;
		Loader loader("..\\datasets\\mnist\\test.txt");

		loader.load(sample);

		std::cout << "the real value is: " << sample.value
			<< "\nthe predict value is: "<< example->forward(sample.img) << std::endl;
	}
	else
	{
		Timer t;
		example->weight_init();
		example->weight_save("..//weight//weight1.w");

		example->backward();
		example->weight_save(weightPath);

		t.pause();
		std::cout << "time: " << t.elapsedTime() << "ms" << std::endl;
	}
	return 0;
}