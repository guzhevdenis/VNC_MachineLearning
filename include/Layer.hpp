//Realization of Network layers 

#ifndef LAYER_H
#define LAYER_H
#include <cstdlib>
#include <iostream>
#include <string>
#include "Operation.hpp"

enum Scaling {
    nearest_neighbour = 1
};

template <typename Type>
class Layer 
{

    public: 
        virtual ~Layer(){};
        virtual void forward(Tensor<Type> &input, Tensor<Type> &output) const = 0;

};

template <typename Type>
class Linear : public Layer<Type>
{
    private:
        Type weights; 
        Type bias;
    public:
    Linear (Tensor<Type> weights, Tensor<Type> bias)
        : weights{weights}, bias{bias}
        {

        }
        ~Linear() = default;
        
    void forward(Tensor<Type> &input, Tensor<Type> &output) const override
    {
        linear_operation(input, output, weights, bias);
    }

        
};

template <typename Type>
class ConvLayer : public Layer <Type>
{
    private:
        Tensor<Type> weights;
        Tensor<Type> bias;

    public:
        ConvLayer(Tensor<Type> weights, Tensor<Type> bias): weights{weights}, bias{bias}
        {
        }
        ~ConvLayer() = default;

        void forward (Tensor<Type> &input, Tensor<Type> &out) const  override
        {
            auto inputShape = input.shape();
            int x = inputShape[0];
            int  y = inputShape[1];
            int z = inputShape[2];

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

            conv2d<Type>(newTensor, out, weights, bias);
        }

};

template <typename Type>
class BatchNorm: public Layer <Type>
{
    private:
        Tensor<Type> weights;
        Tensor<Type> bias;
    public:
    //Нормализация происходит вдоль цветовых каналов (то есть для каждого канала считается свое среднее и вариация)
    BatchNorm(Tensor<Type> weights, Tensor<Type> bias): weights{weights}, bias{bias}
        {

        }
    
    ~BatchNorm() = default;

    void forward (Tensor<Type> &input, Tensor<Type> &output) const  override

    {
        batch_norm<Type>(input, output, weights, bias);
    }


};

template <typename Type>
class Upsampling: public Layer <Type>
{
    private: 
        int scale; 
        Scaling type_of_scaling;
    public: 
        Upsampling (int scale_m): scale(scale_m), type_of_scaling(nearest_neighbour)
        {
            
        }
        ~Upsampling() = default; 

        void forward (Tensor<Type> &input, Tensor<Type> &output) const  override
        {
            if (type_of_scaling == nearest_neighbour)
            {
                upscale_nearest_neighbour(scale, input, output);
            }  
        }
};

//Transposed Convolution -  Повышает размерность изображения
template <typename Type>
class ConvTranspose2D: public Layer <Type>
{
    private:
        Tensor<Type> weights;
        Tensor<Type> bias = 0;
    public:
    
    ConvTranspose2D(Tensor<Type> weights):weights(weights)
        {

        }
    
    ~ConvTranspose2D() = default;

    void forward (Tensor<Type> &input, Tensor<Type> &output) const  override

    {
        conv_transpose_2d<Type>(input, output, weights, bias);
    }


};

#endif