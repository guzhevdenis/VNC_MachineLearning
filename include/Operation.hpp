#include <cmath>
#include "Tensor.hpp"
#include "Activation.hpp"

template <typename Type>
void conv2d (Tensor<Type> &input, Tensor<Type> &ouput, Tensor<Type> const &weight, Tensor<Type> const &bias)
template <typename Type>
void linear_operation (Tensor<Type> &input, Tensor<Type> &output, Tensor<Type> const &weight, Tensor<Type> const &bias);

int to1D(int z, int y, int x, int xSize, int ySize);
int to1D(int f, int z, int y, int x, int xSize, int ySize, int zSize);

int to1D(int z, int y, int x, int xSize, int ySize)
{
    return (ySize * xSize * z) + (xSize * y) + x;
}

int to1D(int f, int z, int y, int x, int xSize, int ySize, int zSize)
{
    return (ySize * zSize * xSize * f) + (ySize * xSize * z) + (xSize * y) + x;
}

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
        
}

template <typename Type>
void conv2d (Tensor<Type> &input, Tensor<Type> &output, Tensor<Type> const &weight, Tensor<Type> const &bias)
{
    auto inputShape = input.shape();
    int i_w = inputShape[0];
    int i_h = inputShape[1];
    int i_f = inputShape[2]; //fmap 

    auto weightShape = weight.shape();
    int w_w = weightShape[0];
    int w_h = weightShape[1];
    int w_f = weightShape[2]; //fmap
    int w_c = weightShape[3]; //channel 

    int o_w = i_w - w_w + 1; 
    int o_h = i_h - w_h + 1; 
    int o_c = w_c; 
    output = Tensor<Type> (o_w, o_h, o_c);
    auto outputShape = output.shape();
    
    //Проверки
    assert(outputShape.size() == 3);
    assert(weightShape.size() == 4);
    assert(i_f == w_f);
    assert(o_c == bias.shape()[0]);

    int n, m, x, y, i, j;

    for (n = 0; n < o_c; n++) // output channel
        {
            for (y = 0; y < o_h; y++) // output y
            {
                for (x = 0; x < o_w; x++) // output x
                {
                    dtype sum = 0;

                    for (m = 0; m < w_f; m++) // kernel fmap
                    {
                        for (j = 0; j < w_h; j++) // kernel y
                        {
                            for (i = 0; i < w_w; i++) // kernel x
                            {
                                
                                Type inputWeight = input[to1D(m, (y + j), (x + i), i_w, i_h)];
                                Type kernelWeight = weight[to1D(n, m, j, i, w_w, w_h, w_f)];

                                sum += inputWeight * kernelWeight;
                            }
                        }
                    }

                    {
                        sum += bias[n];
                        output[to1D(n, y, x, o_w, o_h)] += ReLU(sum);
                    }
                }
            }
        }

    
}
