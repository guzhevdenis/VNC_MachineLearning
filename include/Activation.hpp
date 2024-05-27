//Реализация функций активации 
#ifndef ACTIVATION_H
#define ACTIVATION_H
#include <cmath>

template <typename Type>
Type ReLU(Type input)
{
    return input < 0 ? 0 : input;
}

template <typename Type>
Type Sigmoid(Type input)
{
    return 1/(1+exp(-1*input));
}

template <typename Type>
Type ELU(Type input, float parameter)
{
    return input < 0 ? parameter*input : input;
}
#endif