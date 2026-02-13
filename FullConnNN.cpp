#include "FullConnNN.hpp"
#include "Loader.hpp"
#include "Util.hpp"
#include "Logger.hpp"

extern Logger& logger;

bool FullConnNN::weight_init()
{
	logger.log(logLevel::logINFO, __FILE__, __LINE__, "开始进行模型权重初始化");

	FullConnLayer* tmp_Layer = &input;
	tmp_Layer = tmp_Layer->next;			// 输入层不用初始化权重

	while (tmp_Layer != nullptr)
	{
		tmp_Layer->weight_init();
		tmp_Layer = tmp_Layer->next;
	}

	std::cout << "The network initialization is complete" << std::endl;

	return true;
}

// 存储网络权重参数
// 由于输入层直接接受传入参数，直接从第二层开始存储
bool FullConnNN::weight_save(const std::string& path)
{
	logger.log(logLevel::logINFO, __FILE__, __LINE__, "开始保存模型权重");

	std::vector<std::vector<double>> weightAll;
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
        uint64_t nums = w.size();
        std::fwrite(&nums, sizeof(nums), 1, f);
		std::fwrite(w.data(), sizeof(double), nums, f);
    }
    std::fclose(f);

	std::cout << "weight save successful" <<std::endl;

	return true;
}

bool FullConnNN::weight_load(const std::string &path)
{
	logger.log(logLevel::logINFO, __FILE__, __LINE__, "开始加载模型权重");

	std::vector<std::vector<double>> weightAll;
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
		logger.log(logLevel::logERROR, __FILE__, __LINE__, "权重参数加载失败: 参数n读取失败");
		return false;
	}
    weightAll.resize(n);

	for (auto& w : weightAll) {
        uint64_t nums;
        if (std::fread(&nums, sizeof(nums), 1, f) != 1)
		{
            std::cerr << "weight loading: read nums failed" << std::endl;
			logger.log(logLevel::logERROR, __FILE__, __LINE__, "权重参数加载失败: 参数n读取失败");
			return false;
		}
        w.resize(nums);
		if (std::fread(w.data(), sizeof(double), nums, f) != nums)
		{
			std::cerr << "weight loading: read data failed" << std::endl;
			logger.log(logLevel::logERROR, __FILE__, __LINE__, "权重参数加载失败：当前参数读取失败");
			return false;
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

int FullConnNN::forward(std::vector<double>::const_iterator headIn, std::vector<double>::const_iterator endIn)
{
	FullConnLayer* tmpLayer = &input;

	// 输入层单独计算
	if(tmpLayer->forward(headIn, endIn) != 1)
		return -1;

	while (tmpLayer)
	{
		tmpLayer->forward();
		tmpLayer = tmpLayer->next;
	}

	return output.get_max_output();
}

bool FullConnNN::backward() 
{
	const char *imgPath = "..\\datasets\\mnist\\train-images.idx3-ubyte";
	const char *lblPath = "..\\datasets\\mnist\\train-labels.idx1-ubyte";
	logger.log(logLevel::logINFO, __FILE__, __LINE__, "开始进行反向传播训练");

	std::cout << "begins training" << std::endl;
	double learningStep = 0.001;		// 设置训练步长
	unsigned int epoch = 1;				// 设置训练轮数
	unsigned int countRight = 0;		// forward 正确个数统计
	std::string ratesStr = "";			// 记录每1000个样本训练中的准确率

	std::string outMessage = "本次训练：学习率="+std::to_string(learningStep)
							+", epoch="+std::to_string(epoch);
	logger.log(logLevel::logINFO, __FILE__, __LINE__, outMessage);

	for (unsigned int e = 1; e <= epoch; e++)
	{
		Loader loader(imgPath, lblPath);
		//ProgressBar bar(60000, 50, "progressing", "it");
        

		int n = 0;
		while(n != loader.labels.size())
		{
			++n;
        	loader.load();

			if(this->forward(loader.winBeign, loader.winEnd) 
							== loader.labels[loader.pos])	// 代入训练样本
			{												// 得到中间值,并判断推理结果是否正确
				countRight++;
			}

			if (n % 1000 == 0)								// 迭代一千次后输出
			{												// 并将countRight清零
				std::cout<< "No." << n << ": " << countRight/1000.0 <<std::endl;
				ratesStr += ("\t" + std::to_string(countRight/1000.0));
				countRight = 0;
			}
			
			FullConnLayer* tmp = &output;
			tmp->backward(loader.labels[loader.pos], learningStep);		// 输出层单独计算
			tmp = tmp->prev;

			while (tmp->prev)								// 退出条件：当前层为输入层
			{
				tmp->backward(learningStep);
				tmp = tmp->prev;
			}

			//bar.update(n);
		}
	}
	logger.log(logLevel::logINFO, __FILE__, __LINE__, "训练过程中准确率数据: " + ratesStr);

	return true;
}

double FullConnNN::test()
{
	const char *imgPath = "..//datasets//mnist//t10k-images.idx3-ubyte";
	const char *lblPath = "..//datasets//mnist//t10k-labels.idx1-ubyte";
	logger.log(logLevel::logINFO, __FILE__, __LINE__, "开始进行测试集测试");

	double ret = 0.0;
	Loader loader(imgPath, lblPath);

	int exactNum = 0;
	int allNum = 0;
	int n = 0;
	while(n != loader.labels.size())
	{
		++n;
		loader.load();

		if(this->forward(loader.winBeign, loader.winEnd) 
							== loader.labels[loader.pos])
		{
			exactNum++;
		}
		allNum++;
	}
	ret = static_cast<double>(exactNum)/allNum;

	return ret;
}
