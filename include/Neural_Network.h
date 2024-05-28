#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include <vector>
#include "Layer.hpp"
#include "Tensor.hpp"
template <typename Type>
class Network
{
    private:
        std::vector<Layer<Type> *> layers;
        
    public:
    Network()
    {

    }

    void addLayer (Layer<Type> *layer)
    {
        this->layers.push_back(layer);
    }


    void predict(Tensor<Type> &input, Tensor<Type> &output)
        {
            for (auto layer : layers)
            {
                layer->forward(input, output);
                if (layer != layers.back())
                {
                    input = output;
                }
            }
        }
};
#endif