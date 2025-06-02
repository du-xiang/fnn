#ifndef Loader.H
#define Loader.H

#include <fstream>
#include <vector>
#include <sstream>
#include <string>

// 定义一个结构体用于存储数据集样本的值与内容
struct sample
{
	unsigned int value;
	std::vector<double> img;
};

class Loader()
{
private:
	std::ifstream file;  // 文件流
	std::string line;    // 用于存储当前读取的行
	bool endOfFile;      // 标记是否到达文件末尾

public:

}

#endif // ! Loader.H

