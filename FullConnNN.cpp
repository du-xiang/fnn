#include "FullConnNN.hpp"
#include "Loader.hpp"
#include "Util.hpp"
#include "Logger.hpp"

int FullConnNN::weight_init()
{
	Logger& logger = Logger::getInstance("..//log//log.txt");
	logger.log(logLevel::logINFO, __FILE__, __LINE__, "开始进行模型权重初始化");

	FullConnLayer* tmp_Layer = &input;
	tmp_Layer = tmp_Layer->next;			// 输入层不用初始化权重

	while (tmp_Layer != nullptr)
	{
		tmp_Layer->weight_init();
		tmp_Layer = tmp_Layer->next;
	}

	std::cout << "The network initialization is complete" << std::endl;

	return 1;
}

// 存储网络权重参数
// 由于输入层直接接受传入参数，直接从第二层开始存储
bool FullConnNN::weight_save(const std::string& path)
{
	Logger& logger = Logger::getInstance("..//log//log.txt");
	logger.log(logLevel::logINFO, __FILE__, __LINE__, "开始保存模型权重");

	std::vector<std::vector<std::vector<double>>> weightAll;
	FullConnLayer *tmpLayer = &input;

	while(tmpLayer->next)
	{
		tmpLayer = tmpLayer->next;
		weightAll.push_back(tmpLayer->get_weight());
	}

	FILE *f = std::fopen(path.c_str(), "w+b");
	if (!f) 
	{
		std::cerr << "weight save: fopen failed" << std::endl;
		logger.log(logLevel::logERROR, __FILE__, __LINE__, "模型权重参数保存失败：文件打开失败");
		return false;
	}

	uint64_t n = weightAll.size();
	std::fwrite(&n, sizeof(n), 1, f);

	for (const auto& w : weightAll) 
	{
        uint64_t rows = w.size();
        std::fwrite(&rows, sizeof(rows), 1, f);
        for (const auto& row : w) {
            uint64_t cols = row.size();
            std::fwrite(&cols, sizeof(cols), 1, f);
            std::fwrite(row.data(), sizeof(double), cols, f);
        }
    }
    std::fclose(f);

	std::cout << "weight save successful" <<std::endl;

	return true;
}

bool FullConnNN::weight_load(const std::string &path)
{
	Logger& logger = Logger::getInstance("..//log//log.txt");
	logger.log(logLevel::logINFO, __FILE__, __LINE__, "开始加载模型权重");

	std::vector<std::vector<std::vector<double>>> weightAll;
	FullConnLayer *tmpLayer = &input;

	FILE *f = std::fopen(path.c_str(), "rb");
	if (!f) 
	{
		std::cerr << "weight laoding: fopen failed" << std::endl;
		logger.log(logLevel::logERROR, __FILE__, __LINE__, "权重参数加载失败：文件打开失败");
		return false;
	}

	uint64_t n;
    if (std::fread(&n, sizeof(n), 1, f) != 1)
	{
        std::cerr << "weight loading: read n failed" << std::endl;
		logger.log(logLevel::logERROR, __FILE__, __LINE__, "权重参数加载失败：参数n读取失败");
		return false;
	}
    weightAll.resize(n);

	for (auto& w : weightAll) {
        uint64_t rows;
        if (std::fread(&rows, sizeof(rows), 1, f) != 1)
		{
            std::cerr << "weight loading: read rows failed" << std::endl;
			logger.log(logLevel::logERROR, __FILE__, __LINE__, "权重参数加载失败：参数n读取失败");
			return false;
		}
        w.resize(rows);
        for (auto& row : w) {
            uint64_t cols;
            if (std::fread(&cols, sizeof(cols), 1, f) != 1)
			{
                std::cerr << "weight loading: read cols failed" << std::endl;
				logger.log(logLevel::logERROR, __FILE__, __LINE__, "权重参数加载失败：参数rows读取失败");
				return false;
			}
            row.resize(cols);
            if (std::fread(row.data(), sizeof(double), cols, f) != cols)
			{
                std::cerr << "weight loading: read data failed" << std::endl;
				logger.log(logLevel::logERROR, __FILE__, __LINE__, "权重参数加载失败：当前层参数读取失败");
				return false;
			}
        }
    }
    std::fclose(f);

	int i = 0;
	tmpLayer->weight_init();			// 输入层单独加载，即可视为将其初始化
	while(tmpLayer->next)
	{
		tmpLayer = tmpLayer->next;
		if(tmpLayer->set_weight(weightAll[i]))
			++i;
		else 
		{
			std::cerr << "weight loading: The weight file does not match the model weight" << std::endl;
			logger.log(logLevel::logERROR, __FILE__, __LINE__, "权重参数加载失败：读取参数与网络不匹配");
			return false;
		}
	}

	std::cout << "weight load successful" <<std::endl;

	return true;
}

int FullConnNN::forward(std::vector<double> in)
{
	FullConnLayer* tmpLayer = &input;

	// 输入层单独计算
	if(tmpLayer->forward(in) != 1)
		return -1;

	while (tmpLayer)
	{
		tmpLayer->forward();
		tmpLayer = tmpLayer->next;
	}

	return output.get_max_output();
}

int FullConnNN::backward() 
{
	Logger& logger = Logger::getInstance("..//log//log.txt");
	logger.log(logLevel::logINFO, __FILE__, __LINE__, "开始进行反向传播训练");

	std::cout << "begins training" << std::endl;
	double learningStep = 0.001;		// 设置训练步长
	unsigned int epoch = 1;				// 设置训练轮数
	unsigned int countRight = 0;		// forward 正确个数统计

	std::string outMessage = "本次训练：学习率="+std::to_string(learningStep)
							+", epoch="+std::to_string(epoch);
	logger.log(logLevel::logINFO, __FILE__, __LINE__, outMessage);

	for (unsigned int e = 1; e <= epoch; e++)
	{
        Sample sample;
		Loader loader("..\\datasets\\mnist\\train.txt");
		//ProgressBar bar(60000, 50, "progressing", "it");
        

		int n = 0;
		while(n != 60000)
		{
			++n;
        	while(!loader.load(sample)){};					// 直到读出有效样本

			if(this->forward(sample.img) == sample.value)	// 代入训练样本
				countRight++;								// 得到中间值,并判断推理结果是否正确

			if (n % 1000 == 0)								// 迭代一千次后输出
			{												// 并将countRight清零
				std::cout<< "No." << n << ": " << countRight/1000.0 <<std::endl;
				countRight = 0;
			}
			

			FullConnLayer* tmp = &output;
			tmp->backward(sample.value, learningStep);		// 输出层单独计算
			tmp = tmp->prev;

			while (tmp->prev)								// 退出条件：当前层为输入层
			{
				tmp->backward(learningStep);
				tmp = tmp->prev;
			}
			sample.img.clear();								// 清空img，避免累积

			//bar.update(n);
		}
	}

	return 0;
}

double FullConnNN::test()
{
	Logger& logger = Logger::getInstance("..//log//log.txt");
	logger.log(logLevel::logINFO, __FILE__, __LINE__, "开始进行测试集测试");

	double ret = 0.0;
	Sample sample;
	Loader loader("..\\datasets\\mnist\\test.txt");

	int exactNum = 0;
	int allNum = 0;
	int n = 0;
	while(n != 10000)
	{
		++n;
		while(!loader.load(sample)){};
		
		if(this->forward(sample.img) == sample.value)
		{
			exactNum++;
		}
		allNum++;
		sample.img.clear();
	}
	ret = static_cast<double>(exactNum)/allNum;

	return ret;
}
