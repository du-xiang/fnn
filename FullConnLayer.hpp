#ifndef FullConnLayer_H
#define FullConnLayer_H

#include <vector>
#include <memory>


class FullConnLayer {
private:
	int m_node_num;
	std::vector<std::vector<double>> m_weight;
	std::vector<double> m_output;


public:
	FullConnLayer* prev;
	FullConnLayer* next;

	FullConnLayer();
	FullConnLayer(int n);
	FullConnLayer(int n, FullConnLayer* front_layer);
	//~FullConnLayer();

	int set_node_num(int n);
	int get_node_num() const;
	int weight_init();
	std::vector<double> forward(double (*active_func)(const double& x));
	void display();
};

FullConnLayer::FullConnLayer() {
	m_node_num = 0;

	std::cout << "当前网络层未设置参数！！！" << std::endl;
}

FullConnLayer::FullConnLayer(int n) {
	m_node_num = n;

	m_weight = std::vector<std::vector<double>>(1, std::vector<double>(n));		// 申请 n*1 空间
	m_output = std::vector<double>(n);
}

FullConnLayer::FullConnLayer(int n, FullConnLayer* front_layer) {
	int front_node_num = front_layer->get_node_num();
	m_node_num = n;

	m_weight = std::vector<std::vector<double>>(n,
		std::vector<double>(front_node_num));	// 申请 front_node_num*n 空间
	m_output = std::vector<double>(n);

	//	两层之间链路连接
	front_layer->next = this;
	this->prev = front_layer;
}

int FullConnLayer::set_node_num(int n) {
	m_node_num = n;
	return 1;
}

int FullConnLayer::get_node_num() const {
	return m_node_num;
}

int FullConnLayer::weight_init() {
	for (int i = 0; i < m_weight.size(); i++) {
		for (int j = 0; j < m_weight[i].size(); j++) {
			m_weight[i][j] = 1;
		}
	}

	return 1;
}

std::vector<double> FullConnLayer::forward(double (*active_func)(const double& x)) {
	return m_output;
}

void FullConnLayer::display() {
	std::cout << "本层结点数量：" << this->m_node_num << std::endl;			// 打印各层结点数量

	std::cout << "与上一层间的权重参数：" << std::endl;
	for (int i = 0; i < m_weight.size(); i++) {			// 打印各层间的权重参数
		for (int j = 0; j < m_weight[i].size(); j++) {
			std::cout << m_weight[i][j] << '\t';
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;
}


#endif // !FullConnLayer

