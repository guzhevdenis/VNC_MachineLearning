#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;
#include "Activation.hpp"
#include "Tensor.hpp"
#include "Operation.hpp"




int main(void) {
    

    std::cout<< "Проверка свертки " << std::endl;

    //Входное изображение 5*5*3 
    Tensor<int> Input_A = Tensor<int>(5,5,3); 
    //Тензор фильтров. Размер фильтров 3*3, количество каналов -3, количество фильтров -2
    Tensor<int> Weights_A = Tensor<int>(3,3,3,2);
    //Тензор смещений - равен количеству фильтров
    Tensor<int> Bias_A = Tensor<int>(2); 
    //Выходной тензор - его размеры зависят от размеров входного и фильтров
    Tensor<int> Output_A = Tensor<int>(3,3,2);

    std::vector<int> input_a;
    for (int i = 0; i < 5*5*3; i++)
    {   
        input_a.push_back(i);
    }
    std::vector<int> weights_a;
    for (int i = 0; i < 3*3*3*2; i++)
    {   
        weights_a.push_back(i);
    }
    std::vector<int> bias_a;
    for (int i = 0; i < 2; i++)
    {   
        bias_a.push_back(i);
    }

    Input_A.get_data(input_a);
    Weights_A.get_data(weights_a);
    Bias_A.get_data(bias_a);

    conv2d (Input_A, Output_A, Weights_A, Bias_A);
    std::cout << Output_A << std::endl;


	return 1;

}
