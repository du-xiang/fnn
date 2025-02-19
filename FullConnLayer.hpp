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
	prev = nullptr;
	next = nullptr;

	std::cout << "警告：当前网络层需进行参数设置" << std::endl;
}

FullConnLayer::FullConnLayer(int n) : m_node_num(n), prev(nullptr), next(nullptr) {
	//m_weight = std::vector<std::vector<double>>(1, std::vector<double>(n, 0.5));// 申请1*n 空间
																				// 为使空间连续
																				// 将n*1 变为1*n
	//m_output = std::vector<double>(n, 0.5);
}

FullConnLayer::FullConnLayer(int n, FullConnLayer* front_layer) : m_node_num(n), prev(nullptr) {
	//	两层之间链路连接
	if (front_layer)
	{
		front_layer->next = this;
	}

	//int front_node_num = front_layer->get_node_num();

	//m_weight = std::vector<std::vector<double>>(n,
	//std::vector<double>(front_node_num, 0.5));	// 申请n*front_node_num 权重参数空间
	//m_output = std::vector<double>(n, 0.5);			// 申请n 结果内存空间
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
	std::cout << "本层结点数量：" << get_node_num() << std::endl;			// 打印各层结点数量
	//std::cout << "与上一层间的权重参数：" << std::endl;
	//for (int i = 0; i < m_weight.size(); i++) {			// 打印各层间的权重参数
	//	for (int j = 0; j < m_weight[i].size(); j++) {
	//		std::cout << m_weight[i][j] << '\t';
	//	}
	//	std::cout << std::endl;
	//}

	std::cout << std::endl;
}


#endif // !FullConnLayer

