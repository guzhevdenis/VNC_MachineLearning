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
    auto input = Tensor<int>(224,224,3);
    Tensor<int> output;

    input.randam();

    auto network = Network<int>();
    
    auto *layer1_1 = new ConvLayer<int>(3,3,64);
    network.addLayer(layer1_1);
    network.predict(input, output);

    std::cout << output << std::endl;


	return 1;

}
