#include <iostream>
#include <algorithm>
#include <random>

#include "FullConnLayer.hpp"


int FullConnLayer::get_max_output() const
{
	auto maxIt = std::max_element(layerOutput.begin(), layerOutput.end());

	// 判断最大值是否唯一
	// 不唯一则返回-1
	if((maxIt+1) != layerOutput.end())
	{
		if((*maxIt) != (*std::max_element(maxIt+1, layerOutput.end())))
			return std::distance(layerOutput.begin(), maxIt);
		else
			return -1;
	}
	else
		return std::distance(layerOutput.begin(), maxIt);
}

bool FullConnLayer::weight_init() 
{ 
	std::random_device rd;
	std::mt19937 gen(rd());	
	std::uniform_real_distribution<double> dist(-0.1, 0.1);	
	
	for(int i = 0; i < m_weight.size(); i++)
	{
		for(int j = 0; j < m_weight[i].size(); j++)
		{
			m_weight[i][j] = dist(gen);
		}
	}

	return true;
}

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
		std::cerr << "Error: The input data does not match the number of nodes in the output layer" << std::endl;
		return -1;
	}

	return 1;
}

// 对中间层进行反向传播
// 获取外部数据：front_layer, next_layer
int FullConnLayer::backward(double& learningStep)
{
	double deltaOfWeight = 0;
	std::vector<double> frontLayerOutput = this->prev->layerOutput;
	std::vector<double> nextLayerDelta   = this->next->layerDelta;
	std::vector<std::vector<double>> nextLayerWeight  = this->next->get_weight();

	for(unsigned int i = 0; i < layerOutput.size(); i++)
	{
		double sumOfError = 0.0;
		for(unsigned int j = 0; j < m_node_num_next; j++)
		{
			sumOfError += nextLayerDelta[j]*nextLayerWeight[j][i+1];
		}
		deltaOfWeight = layerOutput[i]*(1-layerOutput[i])*sumOfError;
		for(unsigned int j = 0; j < m_node_num_prev; j++)
		{
			m_weight[i][j] += learningStep*deltaOfWeight*frontLayerOutput[j];
		}
		m_weight[i][m_node_num_prev] += learningStep*deltaOfWeight; // 偏置值单独计算
	}
	return 1;
}

// 对输出层进行反向传播
// 获取外部数据：front_layer, next_layer
int FullConnLayer::backward(unsigned int& valueOfImg, double& learningStep)
{
	double deltaOfWeight = 0;
	std::vector<double> frontLayerOutput = this->prev->layerOutput;

	for(unsigned int i = 0; i < layerOutput.size(); i++)
	{
		int realValue = (i == valueOfImg)? 1 : 0;
		deltaOfWeight = layerOutput[i]*(1-layerOutput[i])*(realValue-layerOutput[i]);
		layerDelta[i] = deltaOfWeight;

		for(unsigned int j = 0; j < m_node_num_prev; j++)
		{
			m_weight[i][j] += learningStep*deltaOfWeight*frontLayerOutput[j];
		}
		m_weight[i][m_node_num_prev] += learningStep*deltaOfWeight; // 偏置值单独计算
	}

	return 1;
}

void FullConnLayer::display() 
{
	std::cout << "No." << get_current_layer() << " layer num of node: " << get_node_num() << std::endl;			// 打印各层结点数量
	std::cout << "Weight parameters between the previous layer: " << std::endl;
	for (unsigned int i = 0; i < m_weight.size(); i++) {			// 打印各层间的权重参数
		for (unsigned int j = 0; j < m_weight[i].size(); j++) {
			std::cout << m_weight[i][j] << '\t';
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;
}
