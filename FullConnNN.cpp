#include "FullConnNN.hpp"

int FullConnNN::weight_init()
{
	FullConnLayer* tmp_Layer = &input;

	while (tmp_Layer != nullptr)
	{
		tmp_Layer->weight_init();
		tmp_Layer = tmp_Layer->next;
	}

	std::cout << "The network initialization is complete" << std::endl;

	return 1;
}

int FullConnNN::forward(std::vector<double> in)
{
	std::cout << "** begins the reasoning process ** " << std::endl;
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
	std::cout << "** begins training **" << std::endl;
	double learningStep = 0.01;				// 设置训练步长
	unsigned int epoch = 10;				// 设置训练轮数

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

	std::cout << "\n** Display of detailed information on network structure: \n" << std::endl;

	while (tmp_Layer != nullptr)
	{
		tmp_Layer->display();
		tmp_Layer = tmp_Layer->next;
	}
}