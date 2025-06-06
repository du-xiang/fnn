#ifndef FullConnLayer_H
#define FullConnLayer_H

#include <vector>
#include <memory>
#include <cmath>


class FullConnLayer {
private:
	unsigned int m_node_num;		// 当前网络层的结点数量
	unsigned int m_current_layer;	// 当前网络层所在的层数
	std::vector<std::vector<double>> m_weight;


public:
	FullConnLayer* prev;
	FullConnLayer* next;
	std::vector<double> layerOutput;

	FullConnLayer();
	FullConnLayer(unsigned int n);
	FullConnLayer(unsigned int n, FullConnLayer* front_layer);
	//~FullConnLayer();

	int set_node_num(unsigned int n);
	unsigned int get_node_num() const;
	int set_current_layer(unsigned int n);
	unsigned int get_current_layer() const;
	int weight_init();
	int forward();
	int forward(std::vector<double>);
	void display();
};

FullConnLayer::FullConnLayer() {
	m_node_num = 0;
	m_current_layer = 0;
	prev = nullptr;
	next = nullptr;

	std::cout << "警告：当前网络层需进行参数设置" << std::endl;
}

FullConnLayer::FullConnLayer(unsigned int n) : m_node_num(n), m_current_layer(1) , 
											prev(nullptr), next(nullptr) {
	m_weight = std::vector<std::vector<double>>(1, std::vector<double>(n));	// 申请1*n 空间
																			// 为使空间连续
																			// 将n*1 变为1*n
	layerOutput = std::vector<double>(n, 1.0);
}

FullConnLayer::FullConnLayer(unsigned int n, FullConnLayer* front_layer) : m_node_num(n) {
	unsigned int front_node_num;

	//	两层之间链路连接
	if (front_layer)
	{
		front_layer->next = this;
		this->prev = front_layer;

		if (front_layer->get_current_layer())
		{
			set_current_layer(front_layer->get_current_layer() + 1);
		}

		front_node_num = front_layer->get_node_num();

		m_weight = std::vector<std::vector<double>>(n,
			std::vector<double>(front_node_num+1, 0.5));	// 申请n*(front_node_num+1) 权重参数空间
		layerOutput = std::vector<double>(n, 0.0);			// 申请n 结果内存空间
	}
	else
	{
		std::cout << "错误：FullConnLayer 传入的front_layer 参数为nullptr" << std::endl;
	}
}

int FullConnLayer::set_node_num(unsigned int n) {
	m_node_num = n;
	return 1;
}

unsigned int FullConnLayer::get_node_num() const {
	return m_node_num;
}

int FullConnLayer::set_current_layer(unsigned int n) {
	m_current_layer = n;
	return 1;
}

unsigned int FullConnLayer::get_current_layer() const {
	return m_current_layer;
}


int FullConnLayer::weight_init() {
	return 1;
}

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

void FullConnLayer::display() {
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


#endif // !FullConnLayer

