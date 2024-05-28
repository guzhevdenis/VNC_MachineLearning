#ifndef OPERATION_H
#define OPERATION_H
#include <cmath>
#include "Tensor.hpp"
#include "Activation.hpp"

template <typename Type>
void conv2d (Tensor<Type> &input, Tensor<Type> &ouput, Tensor<Type> const &weight, Tensor<Type> const &bias);
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

    //Вычисление размеров выходного тензора (с шагом прохождения - 1, без паддингов)
    int o_w = i_w - w_w + 1; 
    int o_h = i_h - w_h + 1; 
    //Глубина свертки - равна количеству фильтров
    int o_c = w_c; 

    //Выходной тензор для свертки 
    output = Tensor<Type> (o_w, o_h, o_c);
    auto outputShape = output.shape();
    
    //Проверки


    int n, //счетчик количества фильтров - глубины выходного тензора 
        m, //счетчик количества каналов изображения -  чаще всего 3 (RGB)
        x, //счетчик ширины выходного тензора - по формуле выше 
        y, //счетчик высоты выходного тензора - по формуле выше 
        i, //счетчик ширины фильтра - чаще всего 3 (равен высоте)
        j; //счетчик высоты фильтра 

    for (n = 0; n < o_c; n++) // проход по количеству фильтров - по глубине выходного тензора 
        {
            for (y = 0; y < o_h; y++) // проход по вы высоте выходного тензора - вычисляется по формуле выше 
            {
                for (x = 0; x < o_w; x++) // проход по ширине выходного тензора - вычисляется по формуле выше 
                { 

                    Type sum = 0;

                    // В следующих трех циклах считается единичный элемент свертки. Каждое значение в фильтре перемножается
                    // на каждое значение в соответсвующем входном изображении и суммируется 
                    for (m = 0; m < w_f; m++) // проход по количеству каналов изображений и фильтров (по дефолту - 3)
                    {
                        for (j = 0; j < w_h; j++) // проход по высоте фильтров 
                        {
                            for (i = 0; i < w_w; i++) // проход по ширине фильтров - по дефолту равны высоте 
                            {

                                Type inputWeight = input[to1D(m, (y + j), (x + i), i_w, i_h)]; //Вычисляется элемент из входного изображения (поскольку в тензоре данные в линейной форме нужно найти необходимый индекс)
                                Type kernelWeight = weight[to1D(n, m, j, i, w_w, w_h, w_f)]; //Вычисляется элемент из фильтра 
                                //Элементы перемножаются и суммируются 
                                sum += inputWeight * kernelWeight;
                            }
                        }
                    }

                    {
                        //Прибавляетс смещение 
                        sum += bias[n];
                        //Полученная сумма записывается в выходной тензор
                        output[to1D(n, y, x, o_w, o_h)] += ReLU(sum);
                    }
                }
            }
        }

    
}
#endif 