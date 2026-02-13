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
	bool weight_init();
	bool weight_save(const std::string& path);
	bool weight_load(const std::string& path);
	int forward(std::vector<double>::const_iterator, std::vector<double>::const_iterator);
	bool backward();
	double test();
};

inline FullConnNN::FullConnNN() : 
	input(784), 
	hidden_1(300, &input), 
	output(10, &hidden_1)
{ }

#endif // !FullConNN_H
