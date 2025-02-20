#ifndef FullConnLayer_H
#define FullConnLayer_H

#include <vector>
#include <memory>


class FullConnLayer {
private:
	int m_node_num;
	std::vector<std::vector<double>> m_weight;


public:
	FullConnLayer* prev;
	FullConnLayer* next;
	std::vector<double> layerOutput;

	FullConnLayer();
	FullConnLayer(int n);
	FullConnLayer(int n, FullConnLayer* front_layer);
	//~FullConnLayer();

	int set_node_num(int n);
	int get_node_num() const;
	int weight_init();
	std::vector<double> forward();
	void display();
};

FullConnLayer::FullConnLayer() {
	m_node_num = 0;
	prev = nullptr;
	next = nullptr;

	std::cout << "���棺��ǰ���������в�������" << std::endl;
}

FullConnLayer::FullConnLayer(int n) : m_node_num(n), prev(nullptr), next(nullptr) {
	m_weight = std::vector<std::vector<double>>(1, std::vector<double>(n));	// ����1*n �ռ�
																			// Ϊʹ�ռ�����
																			// ��n*1 ��Ϊ1*n
	layerOutput = std::vector<double>(n, 1.0);
}

FullConnLayer::FullConnLayer(int n, FullConnLayer* front_layer) : m_node_num(n) {
	int front_node_num;

	//	����֮����·����
	if (front_layer)
	{
		front_layer->next = this;
		this->prev = front_layer;

		front_node_num = front_layer->get_node_num();

		m_weight = std::vector<std::vector<double>>(n,
			std::vector<double>(front_node_num+1, 0.5));	// ����n*(front_node_num+1) Ȩ�ز����ռ�
		layerOutput = std::vector<double>(n, 0.0);				// ����n ����ڴ�ռ�
	}
	else
	{
		std::cout << "����FullConnLayer �����front_layer ����Ϊnullptr" << std::endl;
	}
}

int FullConnLayer::set_node_num(int n) {
	m_node_num = n;
	return 1;
}

int FullConnLayer::get_node_num() const {
	return m_node_num;
}


int FullConnLayer::weight_init() {
	return 1;
}

std::vector<double> FullConnLayer::forward() {
	double tmpOutput;			// ���ڴ洢������̲������м�ֵ
								// ʹ�ü��㲿�ִ��������
	if (this->prev)
	{
		for (int i = 0; i < m_node_num; i++) 
		{
			tmpOutput = m_weight[i][0];
			// ���㹫ʽ w_nx_n +��������+ w_3x_3 + w_2x_2 + w_1x_1 + w0
			for (int j = 0; j < this->prev->get_node_num(); j++)
			{
				tmpOutput += m_weight[i][j+1] * this->prev->layerOutput[j];
			}

			// ʹ��sigmoid ����
			layerOutput[i] = 1.0 / (1.0 + exp(-tmpOutput));
		}
		std::cout << "���������";
		for (int i = 0; i < m_node_num; i++)
			std::cout << layerOutput[i] << '\t';
		std::cout << std::endl;
	}

	return layerOutput;
}

void FullConnLayer::display() {
	std::cout << "������������" << get_node_num() << std::endl;			// ��ӡ����������
	std::cout << "����һ����Ȩ�ز�����" << std::endl;
	for (unsigned int i = 0; i < m_weight.size(); i++) {			// ��ӡ������Ȩ�ز���
		for (unsigned int j = 0; j < m_weight[i].size(); j++) {
			std::cout << m_weight[i][j] << '\t';
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;
}


#endif // !FullConnLayer

