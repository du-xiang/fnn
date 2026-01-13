#ifndef FullConnNN_H
#define FullConnNN_H

#include<iostream>

#include "FullConnLayer.hpp"

class FullConnNN 
{
private:
public:
	FullConnLayer input;
	FullConnLayer hidden_1;
	FullConnLayer output;

	FullConnNN();
	int weight_init();
	bool weight_save(const std::string& path);
	bool weight_load(const std::string& path);
	int forward(std::vector<double> in);
	int backward();
	double test();
};

inline FullConnNN::FullConnNN() : 
	input(784), 
	hidden_1(300, &input), 
	output(10, &hidden_1)
{ }

#endif // !FullConNN_H
