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
    auto input = Tensor<int>(5,5,3);
    auto output = Tensor<int> (5,5,3);

    input.randam();

    auto network = Network<int>();
    
    auto *layer1_1 = new BatchNorm<int>(3);
    network.addLayer(layer1_1);
    network.predict(input, output);

    std::cout << output << std::endl;


	return 1;

}
