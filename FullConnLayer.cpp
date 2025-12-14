#include<iostream>

#include"FullConnLayer.hpp"


int FullConnLayer::weight_init() 
{ return 1;}

// 获取外部数据：front_layer
int FullConnLayer::forward() {
	double tmpOutput;			// 用于存储计算过程产生的中间值
								// 使得计算部分代码更美观
	if (this->prev)
	{
		for (unsigned int i = 0; i < m_node_num; i++) 
		{
			tmpOutput = m_weight[i][0];
			// 计算公式 w_nx_n +····+ w_3x_3 + w_2x_2 + w_1x_1 + w0
			for (unsigned int j = 0; j < this->prev->get_node_num(); j++)
			{
				tmpOutput += m_weight[i][j+1] * this->prev->layerOutput[j];
			}

			// 使用sigmoid 函数
			layerOutput[i] = 1.0 / (1.0 + exp(-tmpOutput));
		}

		std::cout << "第" << get_current_layer() << "层输出：";
		for (unsigned int i = 0; i < m_node_num; i++)
			std::cout << layerOutput[i] << '\t';
		std::cout << std::endl;
	}

	return 1;
}

// 作用于神经网络输入层
// 用于接收数据
int FullConnLayer::forward(std::vector<double> in)
{
	if (layerOutput.size() == in.size())
	{
		layerOutput = in;
	}
	else
	{
		std::cout << "-错误-：输入数据与输出层结点数不匹配" << std::endl;
		return -1;
	}

	std::cout << "第" << get_current_layer() << "层输出：";
	for (unsigned int i = 0; i < m_node_num; i++)
		std::cout << layerOutput[i] << '\t';
	std::cout << std::endl;

	return 1;
}

void FullConnLayer::display() 
{
	std::cout << "第" << get_current_layer() << "层结点数量：" << get_node_num() << std::endl;			// 打印各层结点数量
	std::cout << "与上一层间的权重参数：" << std::endl;
	for (unsigned int i = 0; i < m_weight.size(); i++) {			// 打印各层间的权重参数
		for (unsigned int j = 0; j < m_weight[i].size(); j++) {
			std::cout << m_weight[i][j] << '\t';
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;
}
