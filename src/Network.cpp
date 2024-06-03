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
    auto input = Tensor<double>(1,48);
    auto weights = Tensor<double>(48,9216);
    auto bias = Tensor<double> (9216,1);
    Tensor<double> output;

    input.randam();
    weights.randam();
    //std::cout << input << std::endl;

    auto network = Network<double>();
    
    //Линейный слой
    auto *layer1_1 = new Linear<double>(weights, bias);
    network.addLayer(layer1_1);

    std::vector<int> tensor_shape{512, 6, 3};

    //Преобразование в трехмерный тензор
    auto *layer2_1 = new Unflatten<double>(9216, tensor_shape);
    network.addLayer(layer2_1);

    auto kernels = Tensor<double>(3,3,3,2); //фильтры
    kernels.randam();

    auto bias_kernel = Tensor<double> (2); // по количеству фильтров
    bias_kernel.randam();

    //Сверточный слой
    auto *layer3_1 = new ConvLayer<double> (kernels, bias_kernel);
    network.addLayer(layer3_1);

    Tensor<double> weights_batch = Tensor<double>(2);
    weights_batch.randam();
    Tensor<double> bias_batch = Tensor<double>(2);
    bias_batch.randam();

    //Нормализация + активация
    auto *layer4_1 = new BatchNorm<double> (weights_batch, bias_batch);
    network.addLayer(layer4_1);

    auto kernels_2 = Tensor<double>(2,2,2,2); //фильтры
    kernels_2.randam();

    auto bias_kernel_2 = Tensor<double> (2); // по количеству фильтров
    bias_kernel_2.randam();

    //2 Сверточный слой
    auto *layer5_1 = new ConvLayer<double> (kernels_2, bias_kernel_2);
    network.addLayer(layer5_1);

    Tensor<double> weights_batch_2 = Tensor<double>(2);
    weights_batch_2.randam();
    Tensor<double> bias_batch_2 = Tensor<double>(2);
    bias_batch_2.randam();

    //2 Нормализация + активация
    auto *layer6_1 = new BatchNorm<double> (weights_batch_2, bias_batch_2);
    network.addLayer(layer6_1);


    //Upsampling
    auto *layer7_1 = new Upsampling<double> (2);
    network.addLayer(layer7_1);


    auto kernels_3 = Tensor<double>(2,2,2,2); //фильтры
    kernels_3.randam();

    auto bias_kernel_3 = Tensor<double> (2); // по количеству фильтров
    bias_kernel_3.randam();

    //3 Сверточный слой
    auto *layer8_1 = new ConvLayer<double> (kernels_3, bias_kernel_3);
    network.addLayer(layer8_1);

    Tensor<double> weights_batch_3 = Tensor<double>(2);
    weights_batch_3.randam();
    Tensor<double> bias_batch_3 = Tensor<double>(2);
    bias_batch_3.randam();

    //3 Нормализация + активация
    auto *layer9_1 = new BatchNorm<double> (weights_batch_3, bias_batch_3);
    network.addLayer(layer9_1);
    
    auto kernels_4 = Tensor<double>(2,2,2,2); //фильтры
    kernels_4.randam();

    auto bias_kernel_4 = Tensor<double> (2); // по количеству фильтров
    bias_kernel_4.randam();

    //4 Сверточный слой
    auto *layer10_1 = new ConvLayer<double> (kernels_4, bias_kernel_4);
    network.addLayer(layer10_1);
    
    Tensor<double> weights_batch_4 = Tensor<double>(2);
    weights_batch_4.randam();
    Tensor<double> bias_batch_4 = Tensor<double>(2);
    bias_batch_4.randam();

    //4 Нормализация + активация
    auto *layer11_1 = new BatchNorm<double> (weights_batch_4, bias_batch_4);
    network.addLayer(layer11_1);


    auto kernels_5 = Tensor<double>(2,2,2,2); //фильтры
    kernels_5.randam();

    auto bias_kernel_5 = Tensor<double> (2); // по количеству фильтров
    bias_kernel_5.randam();

  
    //5 Сверточный слой
    auto *layer12_1 = new ConvLayer<double> (kernels_5, bias_kernel_5);
    network.addLayer(layer12_1);


    Tensor<double> weights_batch_5 = Tensor<double>(2);
    weights_batch_5.randam();
    Tensor<double> bias_batch_5 = Tensor<double>(2);
    bias_batch_5.randam();

    //5 Нормализация + активация
    auto *layer13_1 = new BatchNorm<double> (weights_batch_5, bias_batch_5);
    network.addLayer(layer13_1);


    //Upsampling
    auto *layer14_1 = new Upsampling<double> (2);
    network.addLayer(layer14_1);

     auto kernels_6 = Tensor<double>(2,2,2,2); //фильтры
    kernels_6.randam();

    auto bias_kernel_6 = Tensor<double> (2); // по количеству фильтров
    bias_kernel_6.randam();

  
    //6 Сверточный слой
    auto *layer15_1 = new ConvLayer<double> (kernels_6, bias_kernel_6);
    network.addLayer(layer15_1);


    auto kernels_7 = Tensor<double>(2,2,2,2); //фильтры
    kernels_7.randam();

    auto bias_kernel_7 = Tensor<double> (2); // по количеству фильтров
    bias_kernel_7.randam();

  
    //7 Сверточный слой
    auto *layer16_1 = new ConvLayer<double> (kernels_7, bias_kernel_7);
    network.addLayer(layer16_1);


    //Обратный сверточный слой 1

    auto conv_trans_weights = Tensor<double> (1,5);
    conv_trans_weights.randam();

    auto *layer17_1 = new ConvTranspose2D<double>(conv_trans_weights);
    network.addLayer(layer17_1);

    //Обратный сверточный слой 2
    auto conv_trans_weights_2 = Tensor<double> (3,3);
    conv_trans_weights_2.randam();

    auto *layer18_1 = new ConvTranspose2D<double>(conv_trans_weights_2);
    network.addLayer(layer18_1);

    network.predict(input, output);

    std::cout << output.shapeStr() << std::endl;


	return 1;

}
