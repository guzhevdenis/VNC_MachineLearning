#ifndef OPERATION_H
#define OPERATION_H
#include <cmath>
#include "Tensor.hpp"
#include "Activation.hpp"

template <typename Type>
void conv2d (Tensor<Type> &input, Tensor<Type> &ouput, Tensor<Type> const &weight, Tensor<Type> const &bias);
template <typename Type>
void linear_operation (Tensor<Type> &input, Tensor<Type> &output, Tensor<Type> const &weight, Tensor<Type> const &bias);

int to1D(int y, int x, int xSize);
int to1D(int z, int y, int x, int xSize, int ySize);
int to1D(int f, int z, int y, int x, int xSize, int ySize, int zSize);

//для двумерного тензора 
int to1D(int y, int x, int xSize)
{
    return (xSize*y)+x;
}
//для трехмерного тензора
int to1D(int z, int y, int x, int xSize, int ySize)
{
    return (ySize * xSize * z) + (xSize * y) + x;
}

// для четырехмерного тензора 
int to1D(int f, int z, int y, int x, int xSize, int ySize, int zSize)
{
    return (ySize * zSize * xSize * f) + (ySize * xSize * z) + (xSize * y) + x;
}

template <typename Type>
void linear_operation (Tensor<Type> &input, Tensor<Type> &output, Tensor<Type> const &weight, Tensor<Type> const &bias)
{
        //Определяем размерность тензора (размерность будет представлена в виде вектора - в нашем случае это будет одно число 24 или 48)
        auto inputShape = input.shape();

        //Определяем количество данных 
        auto inputLength = input.data().size();

        //Определяем массив весов  - получаем размерность весов (в нашем случае будет 48*9216)
        auto weightShape = weight.shape();

        //Размер выходного тензор должен согласовываться с размером весов (в нашем случае будет 9216)
        int outputSize = weightShape[1];

        output = Tensor<Type>(outputSize);

        
        //Перемножение матриц как вложенные циклы (надо оптимизировать - параллелить)
        for (int i = 0; i < outputSize; i++) // проход по каждому выходному элементу - 9216 штук 
        {
            //Сначала помещаем смещение
            output[i] = bias[i];

            //Применяем итоговую формулу
            for (int k = 0; k < inputLength; k++) // проход по каждому входному элементу 48 штук
            {
                output[i] += input[k] * weight[i + outputSize * k];
            }

            //применение функции активации - может быть добавлена отделным слоем
            output[i] = ReLU(output[i]);
        }
        
}
template <typename Type>
void unflatten (Tensor<Type> &input, Tensor<Type> &output)
{
    std::vector<Type> data = input.data();
    output.get_data(data);
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
                    // на каждое значение в соответствующем входном изображении и суммируется 
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
                        output[to1D(n, y, x, o_w, o_h)] += sum;
                    }
                }
            }
        }

    
}

template <typename Type>
void batch_norm (Tensor<Type> &input, Tensor<Type> &output, Tensor<Type> const &weights, Tensor<Type> const &bias)
{
    int channel_number = input.shape()[2];
    int height = input.shape()[1];
    int width = input.shape()[0];

    for (int i = 0; i < channel_number; i++) //проход по каналам (среднее для каждого канала и считаем)
    {
        Type sum = 0;
        Type mean = 0; 
        Type dispersion = 0;
        double epsilon = 1; //для избежания деления на ноль 

        //sum = bias[i];
        //Вычисление среднего 
        for (int j = 0; j < height; j++)
        {
            for (int k = 0; k < width; k++)
            {
                sum += input[to1D(i,j,k,width,height)]; 
            }  
        }
        
        mean = sum / (width * height);
        

        //Вычисление дисперсии
        for (int j = 0; j < height; j++ )
        {
            for (int k = 0; k < width; k++)
            {
                dispersion += pow((input[to1D(i,j,k,width,height)] - mean), 2); 
            }  
        }

        dispersion = dispersion/ (width * height);

        //Нормализация 
        for (int j = 0; j < height; j++ )
        {
            for (int k = 0; k < width; k++)
            {
               output[to1D(i,j,k,width, height)] = (input[to1D(i,j,k,width, height)] - mean)/ sqrt(dispersion + epsilon); 
            }  
        }

        //Масштабирование и сдвиг 
         for (int j = 0; j < height; j++ )
        {
            for (int k = 0; k < width; k++)
            {
               output[to1D(i,j,k,width, height)] = ReLU(output[to1D(i,j,k,width, height)] * weights[i]+bias[i]); 
            }  
        }

        
    }
}

template <typename Type>
void upscale_nearest_neighbour (int scale, Tensor<Type> &input, Tensor<Type> &output)
{
    int channel_number = input.shape()[2];
    int height = input.shape()[1];
    int width = input.shape()[0];

    //Высота и ширина изображения остается без изменений 
    int o_w = width * scale; 
    int o_h = height * scale; 

    //Количество канал остается без изменений
    int o_c = channel_number; 

    //Выходной тензор после апскейлинга 
    output = Tensor<Type> (o_w, o_h, o_c);
    auto outputShape = output.shape();

    //Потенциальное место для оптимизации - распараллеливание вложенных циклов 
    for (int ch = 0; ch < channel_number; ch++)
    {
        for (int i = 0; i < height; i++) //Проход по высоте входного тензора 
        {
            for (int j = 0; j < width; j++) //Проход по ширине входного тензора 
            {
                //Каждый элемент входного массива a_ij должен продублироваться scale*scale раз в выходном тензоре 
                
                 for (int m = i*scale; m < (i+1)*scale; ++m) //проход по нужным ячейкам выходного тензора m - номер строки
                {
                     for (int k = j*scale; k <(j+1)*scale; ++k) //проход по нужным ячейкам выходного тенхора k - номер столбца
                    {
                  
                           output[to1D(ch, m ,k , width*scale, height*scale)] = input[to1D(ch, i, j, width, height)];
                    }
                }
         
            }
        }
    }

}

//Обратная свертка 
template <typename Type>
void conv_transpose_2d (Tensor<Type> &input, Tensor<Type> &output, Tensor<Type> const &weights, Tensor<Type> const &bias)
{
    //stride = 1, padding = 0
    int height_input = input.shape()[1];
    int width_input = input.shape()[0];
    int height_output = output.shape()[1];
    int width_output = output.shape()[0];

    //Учитываем, что высота и ширина ядра совпадают
    int kernel_size_height = weights.shape()[1];
    int kernel_size_width = weights.shape()[0];

    //Идея позаимствована отсюда
    //https://classic.d2l.ai/chapter_computer-vision/transposed-conv.html

    //Очередное место для оптимизации
    for (int k = 0; k < width_input; k++) //Проход по ширине входного  
    {
            for (int m = 0; m < height_input; m++) //Проход по высоте входного 
            {
                    for (int i = 0; i < kernel_size_width; i++) //Проход по ширине ядра
                    {
                            for (int j = 0; j < kernel_size_height; j++) //Проход по высоте ядра
                            {
                                output[to1D(i+k,j+m,width_output)] += input[to1D(i,j, width_input)]*weights[to1D(k,m, kernel_size_height)];
                            }
                    }
            }
    }

    //ReLU activation
    for (int i = 0; i < width_output; i++) //Проход по ширине выходного тензора
    {
        for (int j = 0; j < height_output; j++) //Проход по высоте выходного тензора 
        {
            output[to1D(j,i,width_output)] = ReLU(output[to1D(j,i,width_output)]);
        }
    }



}
#endif 