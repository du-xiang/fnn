#ifndef FullConnLayer_H
#define FullConnLayer_H

#include <vector>
#include <memory>
#include <cmath>


class FullConnLayer 
{
private:
	unsigned int m_node_num;		// 当前网络层的结点数量
	unsigned int m_node_num_prev;	// 上一层结点数量
	unsigned int m_node_num_next;	// 下一层结点数量
	unsigned int m_current_layer;	// 当前网络层所在的层数
	std::vector<double> m_weight;


public:
	FullConnLayer* prev;
	FullConnLayer* next;
	std::vector<double> layerOutput;
	std::vector<double> layerDelta;

	FullConnLayer() = delete;				// 不允许无参(不声明当前层节点数)构造
	FullConnLayer(unsigned int n);								// 输入层
	FullConnLayer(unsigned int n, FullConnLayer* front_layer);	// 中间+输出层
	//~FullConnLayer();

	bool set_next_ptr(FullConnLayer* n);
	const std::vector<double>& get_layerOutput() const;
	const std::vector<double>& get_layerDelta() const;
	bool set_node_num(unsigned int n);
	const unsigned int& get_node_num() const;
	bool set_node_num_next(unsigned int n);
	const unsigned int& get_node_num_next() const;
	bool set_current_layer(unsigned int n);
	const unsigned int& get_current_layer() const;
	const std::vector<double>& get_weight();
	bool set_weight(std::vector<double> &tmpWeight);
	int get_max_output() const;
	bool weight_init();
	bool forward();
	bool forward(std::vector<double>& imgIn);
	bool forward(std::vector<double>::const_iterator, std::vector<double>::const_iterator);
	bool backward(double& learningStep);
	bool backward(unsigned int& valeOfimg, double& learningStep);
};


inline bool FullConnLayer::set_next_ptr(FullConnLayer* n) 
{ 
	next = n;
	return true;
}

inline const std::vector<double>& FullConnLayer::get_layerOutput() const { return layerOutput;}

inline const std::vector<double>& FullConnLayer::get_layerDelta() const { return layerDelta;}

inline FullConnLayer::FullConnLayer(unsigned int n) : 
	m_node_num(n), 
	m_node_num_prev(0),
	m_node_num_next(0),					// 输入层此参数暂时设置为0
	m_current_layer(1), 
	prev(nullptr),
	next(nullptr) 
{
	m_weight = std::vector<double>(n);	// 申请1*n 空间
	layerOutput = std::vector<double>(n);
	layerDelta  = std::vector<double>(n, 0.0);
}

inline FullConnLayer::FullConnLayer(unsigned int n, FullConnLayer* front_layer) : 
	m_node_num(n) ,
	m_node_num_prev(front_layer->get_node_num()),
	m_node_num_next(0),
    next(nullptr)
{
	//	两层之间链路连接
	if (front_layer)
	{
		front_layer->set_next_ptr(this);
		this->prev = front_layer;

		front_layer->set_node_num_next(n);

		if (front_layer->get_current_layer())
		{
			set_current_layer(front_layer->get_current_layer() + 1);
		}
		else 
		{
			std::cerr << "Error: Unable to obtain the layer number of the upper layer" << std::endl;
		}

		m_weight = std::vector<double>(n*(m_node_num_prev+1));		// 申请n*(front_node_num+1) 权重参数空间
		layerOutput = std::vector<double>(n);						// 申请n 结果内存空间
		layerDelta  = std::vector<double>(n, 0.0); 
	}
	else
	{
		throw std::runtime_error("Error: The front_layer parameter passed to FullConnLayer is nullptr");
	}
}

inline bool FullConnLayer::set_node_num(unsigned int n) 
{
	m_node_num = n;
	return true;
}

inline const unsigned int& FullConnLayer::get_node_num() const { return m_node_num;}

inline bool FullConnLayer::set_current_layer(unsigned int n) 
{
	m_current_layer = n;
	return true;
}

inline const unsigned int& FullConnLayer::get_current_layer() const { return m_current_layer;}

inline const std::vector<double>& FullConnLayer::get_weight() { return m_weight;}

inline bool FullConnLayer::set_weight(std::vector<double> &tmpWeight)
{
	if(this->m_weight.size() == tmpWeight.size())
	{
		this->m_weight = std::move(tmpWeight);
	}
	else return false;

	return true;
}

inline bool FullConnLayer::set_node_num_next(unsigned int n)
{ 
	m_node_num_next = n;
	return true;
}

inline const unsigned int& FullConnLayer::get_node_num_next() const { return m_node_num_next; }

#endif // !FullConnLayer
