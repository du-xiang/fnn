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
	int get_max_output()  const;
	int weight_init();
	int forward();
	int forward(std::vector<double>);
	int backward(double& learningStep);
	int backward(unsigned int& valeOfimg, double& learningStep);
	void display();
};

inline bool FullConnLayer::set_node_num_next(unsigned int n)
{ 
	m_node_num_next = n;
	return true;
}

inline unsigned int FullConnLayer::get_node_num_next() const
{ return m_node_num_next; }

#endif // !FullConnLayer

