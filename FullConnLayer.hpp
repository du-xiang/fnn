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

	FullConnLayer() = delete;	// 不允许无参(不声明当前层节点数)构造
	FullConnLayer(unsigned int n);								// 输入层
	FullConnLayer(unsigned int n, FullConnLayer* front_layer);	// 中间+输出层
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

FullConnLayer::FullConnLayer(unsigned int n) : 
	m_node_num(n), 
	m_current_layer(1) , 
	prev(nullptr),
	next(nullptr) 
{
	m_weight = std::vector<std::vector<double>>(1, std::vector<double>(n));	// 申请1*n 空间
																			// 为使空间连续
																			// 将n*1 变为1*n
	layerOutput = std::vector<double>(n, 1.0);
}

FullConnLayer::FullConnLayer(unsigned int n, FullConnLayer* front_layer) : 
	m_node_num(n) 
{
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
		std::cout << "错误: FullConnLayer 传入的front_layer 参数为nullptr" << std::endl;
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

#endif // !FullConnLayer

