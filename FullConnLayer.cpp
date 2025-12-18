#include<iostream>

#include "FullConnLayer.hpp"


FullConnLayer::FullConnLayer(unsigned int n) : 
	m_node_num(n), 
	m_node_num_prev(0),
	m_node_num_next(0),				// 输入层此参数暂时设置为0
	m_current_layer(1) , 
	prev(nullptr),
	next(nullptr) 
{
	m_weight = std::vector<std::vector<double>>(1, std::vector<double>(n, 0.5));// 申请1*n 空间
																				// 为使空间连续
																				// 将n*1 变为1*n
	layerOutput = std::vector<double>(n, 1.0);
	layerDelta  = std::vector<double>(n, 0.0);
}

// 修改外部数据：front_layer
// 获取外部数据：front_layer
FullConnLayer::FullConnLayer(unsigned int n, FullConnLayer* front_layer) : 
	m_node_num(n) ,
	m_node_num_prev(front_layer->get_node_num()),
	m_node_num_next(0),
    next(nullptr)
{
	//	两层之间链路连接
	if (front_layer)
	{
		front_layer->next = this;
		this->prev = front_layer;

		if (front_layer->get_current_layer())
		{
			set_current_layer(front_layer->get_current_layer() + 1);
		}

		m_weight = std::vector<std::vector<double>>(n,
			std::vector<double>(m_node_num_prev+1, 0.5));	// 申请n*(front_node_num+1) 权重参数空间
		layerOutput = std::vector<double>(n, 0.0);		// 申请n 结果内存空间
		layerDelta  = std::vector<double>(n, 0.0); 
	}
	else
	{
		throw std::runtime_error("Error: The front_layer parameter passed to FullConnLayer is nullptr");
	}
}

int FullConnLayer::set_node_num(unsigned int n) 
{
	m_node_num = n;
	return 1;
}

unsigned int FullConnLayer::get_node_num() const 
{ return m_node_num;}

int FullConnLayer::set_current_layer(unsigned int n) 
{
	m_current_layer = n;
	return 1;
}

unsigned int FullConnLayer::get_current_layer() const 
{ return m_current_layer;}

std::vector<std::vector<double>> FullConnLayer::get_weight()
{ return m_weight;}

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
		throw std::runtime_error("Error: The input data does not match the number of nodes in the output layer");
		return -1;
	}

	return 1;
}

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
			sumOfError += nextLayerDelta[j]*nextLayerWeight[j][1];
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
