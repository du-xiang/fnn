#ifndef Loader.H
#define Loader.H

#include <fstream>
#include <vector>
#include <sstream>
#include <string>

// ����һ���ṹ�����ڴ洢���ݼ�������ֵ������
struct sample
{
	unsigned int value;
	std::vector<double> img;
};

class Loader()
{
private:
	std::ifstream file;  // �ļ���
	std::string line;    // ���ڴ洢��ǰ��ȡ����
	bool endOfFile;      // ����Ƿ񵽴��ļ�ĩβ

public:

}

#endif // ! Loader.H

