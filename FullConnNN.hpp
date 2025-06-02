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

	std::cout << "�����ʼ�����" << std::endl;

	return 1;
}

int FullConnNN::forward(std::vector<double> in)
{
	std::cout << "����������ʼ����������̡�������" << std::endl;
	FullConnLayer* tmpLayer = &input;

	// ����㵥������
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
	std::cout << "����������ʼ����ѵ����������" << std::endl;
	double learningStep = 0.01;			// ����ѵ������
	unsigned int epoch = 1;				// ����ѵ������

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

	std::cout << "����ṹϸ����Ϣչʾ��" << std::endl;

	while (tmp_Layer != nullptr)
	{
		tmp_Layer->display();
		tmp_Layer = tmp_Layer->next;
	}
}

#endif // !FullConNN_H
