#include <cmath>
#include "Tensor.hpp"
#include "Activation.hpp"
//Перемножение матриц. Применение полносвязного слоя 
template <typename Type>
void linear_operation (Tensor<Type> &input, Tensor<Type> &output, Tensor<Type> const &weight, Tensor<Type> const &bias);
template <typename Type>
void linear_operation (Tensor<Type> &input, Tensor<Type> &output, Tensor<Type> const &weight, Tensor<Type> const &bias)
{
        //Определяем размерность тензора (размерность бубдет представлена в виде вектора)
        auto inputShape = input.shape();

        //Определяем количество данных 
        auto inputLength = input.data().size();

        //Определяем массив весов 
        auto weightShape = weight.shape();

        //Размер выходного тензор должен согласовываться с размером весов
        int outputSize = weightShape[1];
        output = Tensor<Type>(outputSize);

        
        //Перемножение матриц как вложенные циклы (надо оптимизировать - параллелить)
        for (int i = 0; i < outputSize; i++)
        {
            //Сначала помещаем смещение
            output[i] = bias[i];

            //Применяем итоговую формулу
            for (int k = 0; k < inputLength; k++)
            {
                output[i] += input[k] * weight[i + outputSize * k];
            }
            output[i] = ReLU(output[i]);
        }
        
        //Тестовые изменения для новой ветки
        for ()
        {

        }
        
}