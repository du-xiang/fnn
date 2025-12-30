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

	return output.get_max_output();
}

int FullConnNN::backward() 
{
	std::cout << "** begins training **" << std::endl;
	double learningStep = 0.1;			// 设置训练步长
	unsigned int epoch = 1;				// 设置训练轮数
	unsigned int countRight = 0;		// forward 正确个数统计

	for (unsigned int e = 1; e <= epoch; e++)
	{
        Sample sample;
		Loader loader("..\\datasets\\mnist\\train.txt");
		//ProgressBar bar(60000, 50, "progressing", "it");
        

		int n = 0;
		while(n != 60000)
		{
			++n;
        	loader.load(sample);

			if(this->forward(sample.img) == sample.value)	// 代入训练样本
				countRight++;								// 得到中间值,并判断推理结果是否正确

			if (n % 1000 == 0)								// 迭代一千次后输出
			{												// 并将countRight清零
				std::cout<< "\nNo." << n << ": " << countRight/1000.0 <<std::endl;
				countRight = 0;
			}
			

			FullConnLayer* tmp = &output;
			tmp->backward(sample.value, learningStep);		// 输出层单独计算
			tmp = tmp->prev;

			while (tmp->prev)								// 退出条件：当前层为输入层
			{
				tmp->backward(learningStep);
				tmp = tmp->prev;
			}
			sample.img.clear();

			//bar.update(n);
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