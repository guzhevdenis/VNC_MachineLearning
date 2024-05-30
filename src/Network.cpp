#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;
#include "Activation.hpp"
#include "Tensor.hpp"
#include "Operation.hpp"
#include "Neural_Network.h"
#include "Layer.hpp"


int main(void) {
    

    std::cout<< "Программа по применению нейронной сети" << std::endl;
    auto input = Tensor<int>(2,2,3);
    auto output = Tensor<int> (6,6,3);

    std::vector<int> a;
    for (int i = 0; i < 2*2*3; i++)
    {
        a.push_back(i);
    }
    input.get_data(a);
    std::cout << input << std::endl;
    auto network = Network<int>();
    
    auto *layer1_1 = new Upsampling<int>(3);
    network.addLayer(layer1_1);
    std::cout << "После upscaling" << std::endl;
    network.predict(input, output);

    std::cout << output << std::endl;


	return 1;

}
