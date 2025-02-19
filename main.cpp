#include<iostream>

#include "FullConnNN.hpp"

int main()
{
	FullConnNN* l = new FullConnNN();
	l->weight_init();
	l->display();

	return 0;
}