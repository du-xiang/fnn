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

#include "httplib.h"

// 训练调试开关
#define TRAINING_DEBUG true
// 使用上一次初始权重
#define USE_PRVE_INIT_WEIGHT true

// 设置全局变量
Logger& logger = Logger::getInstance("..//log//log.txt");
std::shared_ptr<FullConnNN> example(new FullConnNN());
std::string weightPath = "..//weight//weight.w";

bool trainFNN()
{
    Timer t;
    
    if(USE_PRVE_INIT_WEIGHT)
    {
        example->weight_load("..//weight//weight1.w");
        logger.log(logLevel::logINFO, __FILE__, __LINE__, "模型使用上一次的初始化权重");
    }
    else
    {
    example->weight_init();
        logger.log(logLevel::logINFO, __FILE__, __LINE__, "模型权重参数初始化成功");

        example->weight_save("..//weight//weight1.w");
        logger.log(logLevel::logINFO, __FILE__, __LINE__, "成功保持初始化的模型权重参数");
    }

    example->backward();
    logger.log(logLevel::logINFO, __FILE__, __LINE__, "反向传播训练完成");

    example->weight_save(weightPath);
    logger.log(logLevel::logINFO, __FILE__, __LINE__, "模型权重参数存储完成");

    t.pause();
    std::cout << "time: " << t.elapsedTime() << "ms" << std::endl;
	logger.log(logLevel::logINFO, __FILE__, __LINE__, "训练完成。本次训练耗时："+std::to_string(t.elapsedTime())+" ms");

    double rate = example->test();
    logger.log(logLevel::logINFO, __FILE__, __LINE__, "测试完成。正确率: " + std::to_string(rate));

    return true;
}

int predictFNN(std::vector<double> img)
{
    if(example->weight_load(weightPath))
	{
		logger.log(logLevel::logINFO, __FILE__, __LINE__, "成功加载模型权重参数");

		// 为了不把训练中的相关日志信息也写入txt
		// 将此日志信息写入步骤从类成员函数中提取出来
		logger.log(logLevel::logINFO, __FILE__, __LINE__, "开始进行推理");

		return example->forward(img.cbegin(), img.cend());
	}
	else
	{
		logger.log(logLevel::logWARN, __FILE__, __LINE__, "未成功加载模型权重参数，准备重新训练");
        trainFNN();
		return example->forward(img.cbegin(), img.cend());
	}

	return 0;
}

// 空格隔开的数值string解析为vector<double>
static bool str2vector(const std::string &s, std::vector<double> &out) 
{
    out.clear();
    std::istringstream ss(s);
    std::string token;

    while (ss >> token) 
	{
        auto first = token.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        auto last = token.find_last_not_of(" \t\r\n");
        std::string t = token.substr(first, last - first + 1);

        try {
            double v = std::stod(t);
            out.push_back(v);
        } catch (...) {
            return false;
        }
    }
    return true;
}

bool serverInit(httplib::Server &svr)
{
	// 打印所有请求，便于调试
    svr.set_logger([](const httplib::Request &req, const httplib::Response &res){
        std::cout << "[" << req.remote_addr << "] " << req.method << " " << req.path
                  << " -> " << res.status << " (len=" << req.body.size() << ")" << std::endl;
    });

    // 通用的 CORS helper（允许所有来源，调试时方便）
    auto add_cors = [](httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
    };

    // OPTIONS 预检（CORS preflight）
    svr.Options(R"(/process)", [&](const httplib::Request & /*req*/, httplib::Response &res){
        add_cors(res);
        res.status = 200;
        res.set_content("", "text/plain");
    });

    // POST 处理接口
    svr.Post(R"(/process)", [&](const httplib::Request &req, httplib::Response &res){
        add_cors(res);

        std::vector<double> data;
        if (!str2vector(req.body, data)) 
		{
            res.status = 400;
            res.set_content("{\"error\":\"invalid integer data\"}", "application/json");
            return;
        }
        if (data.size() != 784) 
		{
            res.status = 400;
            std::ostringstream err;
            err << "{\"error\":\"expected 784 integers, got " << data.size() << "\"}";
            res.set_content(err.str(), "application/json");
            return;
        }

		int predicValue = predictFNN(data);

        std::ostringstream out;
        out << "{\"result\":" << predicValue << "}";
        res.set_content(out.str(), "application/json");
    });
	return true;
}

int main()
{
	logger.log(logLevel::logINFO, __FILE__, __LINE__, "程序开始运行");

    if (TRAINING_DEBUG)
    {
        trainFNN();
    }
    else
    {
        httplib::Server svr;
        logger.log(logLevel::logINFO, __FILE__, __LINE__, "开启网络连接功能");
        serverInit(svr);
        
        logger.log(logLevel::logINFO, __FILE__, __LINE__, "开启监听,监听127.0.0.1:8080");
        std::cout << "Server listening on http://127.0.0.1:8080\n";
        svr.listen("127.0.0.1", 8080);
    }

	return 0;
}