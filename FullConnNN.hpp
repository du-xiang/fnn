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
	int forward(std::vector<double> in);
	int backward();
	void display();
};

inline FullConnNN::FullConnNN() : 
	input(784), 
	hidden_1(3, &input), 
	output(300, &hidden_1)
{ }

#endif // !FullConNN_H
