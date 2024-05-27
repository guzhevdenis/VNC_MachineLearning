//Realization of Network layers 

#include <cstdlib>
#include <iostream>
#include <string>
#include "Operation.hpp"

template <typename Type>
class Layer 
{
    private:
        std::string name;

    public: 
        virtual ~Layer(){};

        virtual std::string getName() const = 0; 
        virtual void forward(Type &input, Type &output) const = 0;

};

template <typename Type>
class Linear : public Layer<Type>
{
    private:
        Type weight; 
        Type bias;
        int inputSize; 
        int outputSize;

    public:
    Linear (int inputSize, int outputSize)
        : inputSize{inputSize}, outputSize{outputSize}
        {
            weight = 0;
            bias = 0;
        }
        ~Linear() default;
        
    void forward(Type &input, Type &output) const override
    {
        linear_operation(input, output, weight, bias);
    }

        
};

template <typename Type>
class ConvLayer : public Layer <Type>
{
    private:
        Tensor<Type> weight;
        Tensor<Type> bias;

    public:
        ConvLayer(int fmapSize, int kernelSize, int channelSize)
        {
            weight = Tensor<Type>(kernelSize, kernelSize, fmapSize, channelSize);
            weight.randam();
            bias = Tensor<Type>(channelSize);
            bias.randam();
        }
        ~ConvLayer() = default;

        void forward (Tensor<Type> &input, Tensor<Type> &out) const override 
        {
            auto inputShape input.shape();
            int x = iputShape[0], y = inputShape[1], z = inputShape[2];

            Tensor<Type> newTensor(x+2, y+2, z);
            auto newShape = newTensor.shape();

            for (int k = 0; k < z; k++)
            {
                for (int j = 0; j < y; j++)
                {
                    for(int i = 0; i < x; i++)
                    {
                        int newId = to1D(k, j+1, i+1, x+2, y);
                        int oldId = to1D(k, j, i, x, y);
                        newTensor[newId] = input[oldId];
                    }
                }
            }
            cond2d<Type>(newTensor, out, weight, bias);
        }

}