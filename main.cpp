#include <iostream>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <cstdio>
#include <cstdint>
#include <memory>

#include "FullConnNN.hpp"
#include "Util.hpp"
#include "Loader.hpp"
#include "Logger.hpp"

bool predictFNN(std::vector<double> img)
{
	Logger& logger = Logger::getInstance("..//log//log.txt");
	logger.log(logLevel::logINFO, __FILE__, __LINE__, "程序开始运行");

	std::string weightPath = "..//weight//weight.w";
	std::shared_ptr<FullConnNN> example(new FullConnNN());

	if(example->weight_load(weightPath))
	{
		logger.log(logLevel::logINFO, __FILE__, __LINE__, "成功加载模型权重参数");

		// 为了不把训练中的相关日志信息也写入txt
		// 将此日志信息写入步骤从类成员函数中提取出来
		logger.log(logLevel::logINFO, __FILE__, __LINE__, "开始进行推理");


		//Sample sample;
		//Loader loader("..\\datasets\\mnist\\test.txt");

		//loader.load(sample);

		//std::cout << "the real value is: " << sample.value
		//	<< "\nthe predict value is: "<< example->forward(sample.img) << std::endl;
	}
	else
	{
		logger.log(logLevel::logWARN, __FILE__, __LINE__, "未成功加载模型权重参数，准备重新训练");

		Timer t;
		example->weight_init();
		logger.log(logLevel::logINFO, __FILE__, __LINE__, "模型权重参数初始化成功");

		example->weight_save("..//weight//weight1.w");
		logger.log(logLevel::logINFO, __FILE__, __LINE__, "成功保持初始化的模型权重参数");

		example->backward();
		logger.log(logLevel::logINFO, __FILE__, __LINE__, "反向传播训练完成");

		example->weight_save(weightPath);
		logger.log(logLevel::logINFO, __FILE__, __LINE__, "模型权重参数存储完成");

		t.pause();
		std::cout << "time: " << t.elapsedTime() << "ms" << std::endl;
		logger.log(logLevel::logINFO, __FILE__, __LINE__, "训练完成。本次训练耗时："+std::to_string(t.elapsedTime())+" ms");
	}

	return true;
}

int main()
{
	std::vector<double> img;
	predictFNN(img);

	return 0;
}