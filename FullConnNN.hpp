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
	int forward(std::vector<double> in);
	int backward();
	void display();
};

FullConnNN::FullConnNN() : input(2), hidden_1(3, &input), output(2, &hidden_1)
{

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

int FullConnNN::forward(std::vector<double> in)
{
	std::cout << "————开始进行推理过程————" << std::endl;
	FullConnLayer* tmpLayer = &input;

	// 输入层单独计算
	if(tmpLayer->forward(in) != 1)
		return -1;

	while (tmpLayer)
	{
		tmpLayer->forward();
		tmpLayer = tmpLayer->next;
	}

	return 1;
}

int FullConnNN::backward() 
{
	std::cout << "————开始进行训练————" << std::endl;
	double learningStep = 0.01;			// 设置训练步长
	unsigned int epoch = 1;				// 设置训练轮数

	for (unsigned int e = 1; e <= epoch; e++)
	{
		std::vector<double> in(2, 10);
		this->forward(in);

		FullConnLayer* tmp = &output;
		while (tmp->prev)
		{
			

			tmp = tmp->prev;
		}
	}

	return 0;
}

void FullConnNN::display()
{
	FullConnLayer* tmp_Layer = &input;

	std::cout << "网络结构细节信息展示：" << std::endl;

	while (tmp_Layer != nullptr)
	{
		tmp_Layer->display();
		tmp_Layer = tmp_Layer->next;
	}
}

#endif // !FullConNN_H
