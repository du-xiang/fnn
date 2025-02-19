#ifndef FullConnNN_H
#define FullConnNN_H

#include<iostream>

#include "FullConnLayer.hpp"

class FullConnNN {
private:
public:
	FullConnLayer input;
	FullConnLayer hidden_1;
	FullConnLayer output;

	FullConnNN();

	int weight_init();
	int forward();
	void display();
};

FullConnNN::FullConnNN()
{
	FullConnLayer input(5);
	FullConnLayer hidden_1(4, &input);
	FullConnLayer output(3, &hidden_1);
}

int FullConnNN::weight_init()
{
	FullConnLayer* tmp_Layer = &input;

	while (tmp_Layer != nullptr)
	{
		tmp_Layer->weight_init();
		tmp_Layer = tmp_Layer->next;
	}

	std::cout << "网络初始化完成" << std::endl;

	return 1;
}

int FullConnNN::forward()
{
	return 1;
}

void FullConnNN::display()
{
	FullConnLayer* tmp_Layer = &input;

	std::cout << "Network structure detail information display:" << std::endl;

	while (tmp_Layer != nullptr)
	{
		tmp_Layer->display();
		tmp_Layer = tmp_Layer->next;
	}
}

#endif // !FullConNN_H
