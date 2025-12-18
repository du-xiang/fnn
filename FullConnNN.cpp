#include "FullConnNN.hpp"
#include "Loader.hpp"
#include "Util.hpp"

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
	double learningStep = 0.01;			// 设置训练步长
	unsigned int epoch = 1;				// 设置训练轮数

	for (unsigned int e = 1; e <= epoch; e++)
	{
        Sample sample;
		Loader loader("..\\datasets\\mnist\\train.txt");
		ProgressBar bar(60000, 50, "progressing", "it");
        
        // 此部分需批次内按样本重复，以此为退出条件
		// 目前使用1000样本进行性能测试
		int n = 0;
		while(n != 60000)
		{
			++n;
        	loader.load(sample);

			this->forward(sample.img);  // 代入训练样本
                                        // 得到中间值

			FullConnLayer* tmp = &output;
        	tmp->backward(sample.value, learningStep);	// 输出层单独计算
        	tmp = tmp->prev;

			while (tmp->prev)				// 退出条件：当前层为输入层
			{
				tmp->backward(learningStep);
				tmp = tmp->prev;
			}
			sample.img.clear();

			bar.update(n);
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