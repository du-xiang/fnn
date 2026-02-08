#include <iostream>
#include <algorithm>
#include <random>

#include "FullConnLayer.hpp"
#include "Logger.hpp"


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
		m_weight[i] = dist(gen);
	}

	return true;
}

// 获取外部数据：front_layer
int FullConnLayer::forward() {
	double tmpOutput;			// 用于存储计算过程产生的中间值
								// 使得计算部分代码更美观
	const std::vector<double>& frontLayerOutput = this->prev->get_layerOutput();

	if (this->prev)
	{
		for (unsigned int i = 0; i < m_node_num; i++) 
		{
			tmpOutput = m_weight[i*m_node_num_prev];
			// 计算公式 w_nx_n +····+ w_3x_3 + w_2x_2 + w_1x_1 + w0
			for (unsigned int j = 0; j < m_node_num_prev; j++)
			{
				tmpOutput += m_weight[i*(m_node_num_prev+1)+j+1] * frontLayerOutput[j];
			}

			// 使用sigmoid 函数
			layerOutput[i] = 1.0 / (1.0 + exp(-tmpOutput));
		}
	}

	return 1;
}

// 作用于神经网络输入层
// 用于接收数据
int FullConnLayer::forward(std::vector<double>::const_iterator headIn, std::vector<double>::const_iterator endIn)
{
	Logger& logger = Logger::getInstance("..//log//log.txt");

	if (layerOutput.size() == endIn-headIn)
	{
		std::vector<double>::const_iterator tmpIn = headIn;
		for(int i = 0; i < layerOutput.size(); i++)
		{
			layerOutput[i] = *tmpIn;
			++tmpIn;
		}
	}
	else
	{
		std::cerr << "Error: The input data does not match the number of nodes in the input layer" << std::endl;
		logger.log(logLevel::logERROR, __FILE__, __LINE__, "输入数据与输入层结点数量大小不匹配");
		return -1;
	}

	return 1;
}

// 对中间层进行反向传播
int FullConnLayer::backward(double& learningStep)
{
	double deltaOfWeight = 0;
	const std::vector<double>& frontLayerOutput = this->prev->get_layerOutput();
	const std::vector<double>& nextLayerDelta   = this->next->get_layerDelta();
	std::vector<double> nextLayerWeight  = this->next->get_weight();

	for(unsigned int i = 0; i < layerOutput.size(); i++)
	{
		double sumOfError = 0.0;
		for(unsigned int j = 0; j < m_node_num_next; j++)
		{
			sumOfError += nextLayerDelta[j]*nextLayerWeight[j*(m_node_num+1)+i+1];
		}
		deltaOfWeight = layerOutput[i]*(1-layerOutput[i])*sumOfError;
		for(unsigned int j = 0; j < m_node_num_prev; j++)
		{
			m_weight[i*(m_node_num_prev+1)+j] += learningStep*deltaOfWeight*frontLayerOutput[j];
		}
		m_weight[i*(m_node_num_prev+1)+m_node_num_prev] += learningStep*deltaOfWeight; // 偏置值单独计算
	}
	return 1;
}

// 对输出层进行反向传播
int FullConnLayer::backward(unsigned int& valueOfImg, double& learningStep)
{
	double deltaOfWeight = 0;
	const std::vector<double>& frontLayerOutput = this->prev->get_layerOutput();

	for(unsigned int i = 0; i < layerOutput.size(); i++)
	{
		int realValue = (i == valueOfImg)? 1 : 0;
		deltaOfWeight = layerOutput[i]*(1-layerOutput[i])*(realValue-layerOutput[i]);
		layerDelta[i] = deltaOfWeight;

		for(unsigned int j = 0; j < m_node_num_prev; j++)
		{
			m_weight[i*(m_node_num_prev+1)+j] += learningStep*deltaOfWeight*frontLayerOutput[j];
		}
		m_weight[i*(m_node_num_prev+1)+m_node_num_prev] += learningStep*deltaOfWeight; // 偏置值单独计算
	}

	return 1;
}
