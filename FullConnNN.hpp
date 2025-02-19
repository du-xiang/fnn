#ifndef FullConnNN_H
#define FullConnNN_H

#include<iostream>

#include "FullConnLayer.hpp"
#include "util.cpp"

class FullConnNN {
private:
public:
	std::shared_ptr<FullConnLayer> input;
	std::shared_ptr<FullConnLayer> hidden_1;
	std::shared_ptr<FullConnLayer> output;

	FullConnNN();

	int weight_init();
	int forward();
	void display();
};

FullConnNN::FullConnNN()
{
	input = std::make_shared<FullConnLayer>(5);
	hidden_1 = std::make_shared<FullConnLayer>(4, input);
	output = std::make_shared<FullConnLayer>(3, hidden_1);
}

int FullConnNN::weight_init()
{
	std::shared_ptr<FullConnLayer> tmp_Layer(input);

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
	std::shared_ptr<FullConnLayer> tmp_Layer(input);

	std::cout << "Network structure detail information display:" << std::endl;

	while (tmp_Layer != nullptr)
	{
		tmp_Layer->display();
		tmp_Layer = tmp_Layer->next;
	}
}

#endif // !FullConNN_H
