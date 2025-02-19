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

	std::cout << "���棺��ǰ���������в�������" << std::endl;
}

FullConnLayer::FullConnLayer(int n) : m_node_num(n), prev(nullptr), next(nullptr) {
	//m_weight = std::vector<std::vector<double>>(1, std::vector<double>(n, 0.5));// ����1*n �ռ�
																				// Ϊʹ�ռ�����
																				// ��n*1 ��Ϊ1*n
	//m_output = std::vector<double>(n, 0.5);
}

FullConnLayer::FullConnLayer(int n, FullConnLayer* front_layer) : m_node_num(n), prev(nullptr) {
	//	����֮����·����
	if (front_layer)
	{
		front_layer->next = this;
	}

	//int front_node_num = front_layer->get_node_num();

	//m_weight = std::vector<std::vector<double>>(n,
	//std::vector<double>(front_node_num, 0.5));	// ����n*front_node_num Ȩ�ز����ռ�
	//m_output = std::vector<double>(n, 0.5);			// ����n ����ڴ�ռ�
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
	std::cout << "������������" << get_node_num() << std::endl;			// ��ӡ����������
	//std::cout << "����һ����Ȩ�ز�����" << std::endl;
	//for (int i = 0; i < m_weight.size(); i++) {			// ��ӡ������Ȩ�ز���
	//	for (int j = 0; j < m_weight[i].size(); j++) {
	//		std::cout << m_weight[i][j] << '\t';
	//	}
	//	std::cout << std::endl;
	//}

	std::cout << std::endl;
}


#endif // !FullConnLayer

