#ifndef NN_H
#define NN_H
#include "matrix.h"


namespace nn{
    enum nonlinear{Relu, Sigmoid, LogSoftmax, linear_out};

    class Module{
        

    };

    class Linear{
    public:
        bool bias;
        base_matrix *weight;
        matrix (*activation)(const matrix &v);
        Linear(int in_dim, int out_dim, int nonlinear, bool bias);
        ~Linear();
        matrix forward(const matrix &data);
    };

}

#endif
