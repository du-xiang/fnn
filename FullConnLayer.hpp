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

#endif // !FullConnLayer

