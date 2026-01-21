#ifndef FullConnLayer_H
#define FullConnLayer_H

#include <vector>
#include <memory>
#include <cmath>


class FullConnLayer {
private:
	unsigned int m_node_num;		// 当前网络层的结点数量
	unsigned int m_node_num_prev;	// 上一层结点数量
	unsigned int m_node_num_next;	// 下一层结点数量
	unsigned int m_current_layer;	// 当前网络层所在的层数
	std::vector<std::vector<double>> m_weight;


public:
	FullConnLayer* prev;
	FullConnLayer* next;
	std::vector<double> layerOutput;
	std::vector<double> layerDelta;

	FullConnLayer() = delete;		// 不允许无参(不声明当前层节点数)构造
	FullConnLayer(unsigned int n);								// 输入层
	FullConnLayer(unsigned int n, FullConnLayer* front_layer);	// 中间+输出层
	//~FullConnLayer();

	int set_node_num(unsigned int n);
	unsigned int get_node_num() const;
	bool set_node_num_next(unsigned int n);
	unsigned int get_node_num_next() const;
	int set_current_layer(unsigned int n);
	unsigned int get_current_layer() const;
	std::vector<std::vector<double>> get_weight();
	bool set_weight(std::vector<std::vector<double>> &tmpWeight);
	int get_max_output()  const;
	bool weight_init();
	int forward();
	int forward(std::vector<double>::const_iterator, std::vector<double>::const_iterator);
	int backward(double& learningStep);
	int backward(unsigned int& valeOfimg, double& learningStep);
};

inline FullConnLayer::FullConnLayer(unsigned int n) : 
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
inline FullConnLayer::FullConnLayer(unsigned int n, FullConnLayer* front_layer) : 
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

		front_layer->set_node_num_next(n);

		if (front_layer->get_current_layer())
		{
			set_current_layer(front_layer->get_current_layer() + 1);
		}

		m_weight = std::vector<std::vector<double>>(n,
			std::vector<double>(m_node_num_prev+1, 0.5));	// 申请n*(front_node_num+1) 权重参数空间
		layerOutput = std::vector<double>(n, 0.0);			// 申请n 结果内存空间
		layerDelta  = std::vector<double>(n, 0.0); 
	}
	else
	{
		throw std::runtime_error("Error: The front_layer parameter passed to FullConnLayer is nullptr");
	}
}

inline int FullConnLayer::set_node_num(unsigned int n) 
{
	m_node_num = n;
	return 1;
}

inline unsigned int FullConnLayer::get_node_num() const 
{ return m_node_num;}

inline int FullConnLayer::set_current_layer(unsigned int n) 
{
	m_current_layer = n;
	return 1;
}

inline unsigned int FullConnLayer::get_current_layer() const 
{ return m_current_layer;}

inline std::vector<std::vector<double>> FullConnLayer::get_weight()
{ return m_weight;}

inline bool FullConnLayer::set_weight(std::vector<std::vector<double>> &tmpWeight)
{
	if(this->m_weight.size() == tmpWeight.size())
	{
		if(tmpWeight[0].size() == this->m_weight[0].size())
		{
			this->m_weight = tmpWeight;
		}
		else return false;
	}
	else return false;

	return true;
}

inline bool FullConnLayer::set_node_num_next(unsigned int n)
{ 
	m_node_num_next = n;
	return true;
}

inline unsigned int FullConnLayer::get_node_num_next() const
{ return m_node_num_next; }

#endif // !FullConnLayer
