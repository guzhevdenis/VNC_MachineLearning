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

        
}