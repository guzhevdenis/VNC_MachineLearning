#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;
#include "Activation.hpp"
#include "Tensor.hpp"
#include "Operation.hpp"




int main(void) {
    

    Tensor<int> Input = Tensor<int>(3,1);
    Tensor<int> Weights = Tensor<int>(6,3);
    Tensor<int> Bias = Tensor<int>(3,1);
    Tensor<int> Output = Tensor<int>(6,1);

    std::vector<int> input {1,2,3};
    Input.get_data(input);

    std::cout << Input << std::endl;

    std::vector<int> weights; 
    for (int i = 0; i < 18; i++)
    {   
        weights.push_back(i);
    }

    Weights.get_data(weights);
    
    linear_operation(Input, Output, Weights, Bias);
    std::cout << Output <<std::endl;

	return 1;

}
