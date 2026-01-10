#ifndef Loader_H
#define Loader_H

#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <stdexcept>

// 定义一个结构体用于存储数据集样本的值与内容
struct Sample
{
	unsigned int value;
	std::vector<double> img;
};

class Loader
{
private:
	std::ifstream file_;  // 文件流

public:
	Loader() = delete;
	explicit Loader(const std::string& filePath);
	~Loader();
	bool load(Sample& s);

};

inline Loader::Loader(const std::string& filePath)
{
	file_.open(filePath);

	if(!file_.is_open())
	{
		throw std::runtime_error("Cannot open file: " + filePath);
	}
}

inline Loader::~Loader()
{
	if(file_.is_open())
	{
		file_.close();
	}
}

inline bool Loader::load(Sample& s)
{
	std::string line;

	if(!std::getline(file_, line))		// 判断是否到达文件末尾
		return false;

	std::istringstream iss(line);
	double val;
	std::size_t cnt = 0;

	iss >> s.value;
	while (iss >> val)
	{
		s.img.push_back(val);
		++cnt;
	}

	if(cnt != 784)
	{
		std::cerr << "Line does not contain exactly 784 numbers!" << std::endl;	// 错误时会跳过此样本，所以无需异常处理
		return false;
	}

	return true;
}

#endif // ! Loader_H
