#include <vector>
#include <iostream>
#include <sstream>
#ifndef TENSOR_H
#define TENSOR_H

 template <typename Type>
    class Tensor
    {

    private:
        //Данные тензора 
        std::vector<Type> mData;
        //Размер тензора представлен  вектором 
        std::vector<int> mShape;

    public:
        //Пустой конструктор
        Tensor()
        {
        }
        //Конструкторы копирования и присваивания
        Tensor(Tensor &other) : mData{other.mData}, mShape{other.mShape} {}
        Tensor(Tensor &&other) : mData{std::move(other.mData)}, mShape{std::move(other.mShape)} {}

        //Конструкторы тензора через его размер 

        //Одномерный тензор - вектор 
        Tensor(int a)
        {
            mData.resize(a);
            mShape.push_back(a);
        }

        //Матрица
        Tensor(int a, int b)
        {
            mData.resize(a * b);
            this->mShape.push_back(a);
            this->mShape.push_back(b);
        }

        //Трехмерная матрица
        Tensor(int a, int b, int c)
        {
            mData.resize(a * b * c);
            this->mShape.push_back(a);
            this->mShape.push_back(b);
            this->mShape.push_back(c);
        }

        //и т.д. 
        Tensor(int a, int b, int c, int d)
        {
            mData.resize(a * b * c * d);
            this->mShape.push_back(a);
            this->mShape.push_back(b);
            this->mShape.push_back(c);
            this->mShape.push_back(d);
        }

        Tensor &operator=(Tensor<Type> &tensor)
        {
            //умное перемещение для эффективности копирования
            mShape = std::move(tensor.mShape);
            mData = std::move(tensor.mData);
            return *this;
        }

        Tensor &operator=(Tensor<Type> &&tensor)
        {
            mShape = std::move(tensor.mShape);
            mData = std::move(tensor.mData);
            return *this;
        }

        //Вычисление элемента по индексу (индекс одномерный)
        Type &operator[](int idx)
        {
            return mData[idx];
        }

        Type const &operator[](int idx) const
        {
            return mData[idx];
        }

        ~Tensor() {}
        

        //Геттер внутренних данных 
        std::vector<Type> &data()
        {
            return this->mData;
        }

        std::vector<Type> data() const
        {
            return this->mData;
        }
        
        //Геттер размера 
        std::vector<int> shape()
        {
            return this->mShape;
        }

        std::vector<int> shape() const
        {
            return this->mShape;
        }

        //Вывод размера в виде строки
        std::string shapeStr() const
        {
            std::stringstream ss;
            for (int i = 0; i < this->mShape.size(); i++)
            {
                ss << mShape[i] << " ";
            }
            return ss.str();
        }

        std::string shapeStr()
        {
            std::stringstream ss;
            for (int i = 0; i < this->mShape.size(); i++)
            {
                ss << mShape[i] << " ";
            }
            return ss.str();
        }

        std::string str() const
        {
            if (mShape.size() == 4)
                return strND(4);
            if (mShape.size() == 3)
                return strND(3);
            if (mShape.size() == 2)
                return strND(2);
            return strND(1);
        }

        //Вывод тензора на экран в удобной форме 
        std::string strND(int N) const
        {
            std::stringstream ss;
            for (int i = 0; i < mData.size(); i++)
            {
                if (N == 4)
                {
                    if (i % (mShape[0] * mShape[1] * mShape[2]) == 0 && i != 0)
                    {
                        ss << std::endl;
                    }
                }
                if (N >= 3)
                {
                    if (i % (mShape[0] * mShape[1]) == 0 && i != 0)
                    {
                        ss << std::endl;
                    }
                }
                if (N >= 2)
                {
                    if (i % mShape[0] == 0 && i != 0)
                    {
                        ss << std::endl;
                    }
                }
                ss << mData[i] << " ";
            }
            return ss.str();
        }

        //Определение оператора вывода на экран для тензора 
        friend std::ostream &operator<<(std::ostream &inOStream, Tensor const &tensor)
        {
            inOStream << tensor.str();
            return inOStream;
        }

        //Задание тензору рандомных значений
    
        void randam()
        {
            for (int i = 0; i < mData.size(); i++)
            {
                mData[i] = (Type)std::rand();
            }
        }

        //Функция загрузки данных в массив - может быть оптимизирована
        void get_data(std::vector<Type> input_data)
        {
             for (int i = 0; i < mData.size(); i++)
            {
                mData[i] = input_data[i];
            }
        }
    };
#endif