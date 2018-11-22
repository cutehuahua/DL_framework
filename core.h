#ifndef TENSOR_H
#define TENSOR_H
#include <iostream>
#include <math.h>
#include <random>
#include <cstdlib>
#include "core.h"

#define base_matrix_apply(mat, op_code)                      \
    for (int i = 0; i < mat.height * mat.width; i++){ \
        op_code}

class base_matrix{
public:
    float *value;
    int height, width;
    bool reference;

    ~base_matrix();
    base_matrix();
    base_matrix(int h, int w);
    base_matrix(int h, int w, bool zero);
    base_matrix(int h, int w, float *value, bool ref);
    base_matrix(const base_matrix &mat);
    base_matrix(const base_matrix *mat);
    float at(int h, int w) const;
    void row_assign(const base_matrix &v, int row);

    //element-wise operator
    base_matrix operator+ (const float &v);
    base_matrix operator- (const float &v);
    base_matrix operator* (const float &v);
    base_matrix operator/ (const float &v);
    base_matrix operator+ (const base_matrix &v);
    base_matrix operator- (const base_matrix &v);
    base_matrix operator* (const base_matrix &v);
    base_matrix operator/ (const base_matrix &v);
    void operator-= (const base_matrix &v);
    void operator= (const base_matrix &v);

    //base_matrix product
    base_matrix _matmul (const base_matrix &mat);
    base_matrix T();
    base_matrix transpose();
    float max();

    friend float row_sum(const base_matrix &mat, int row);
    friend std::ostream& operator<< (std::ostream& out, const base_matrix &mat);
    friend base_matrix sigmoid(const base_matrix &v);
    friend base_matrix relu(const base_matrix &v);
    friend base_matrix pow(const base_matrix &v, float p);
    friend base_matrix exp(const base_matrix &v);

    friend float sum(const base_matrix &v);
    friend base_matrix cat(const base_matrix &a, const base_matrix &b, int dim);
    friend base_matrix logSoftmax(const base_matrix &v);
};


class node{
public:
    node *next, *previous;
    base_matrix *input_data, *weight, *gradient;
    bool bias;
    void operator= (const node &v);
    node();
    ~node();
};



class matrix : public base_matrix{
public:
    node *tracker;

    matrix();
    ~matrix();
    matrix(int h, int w);
    matrix(int h, int w, float *value, bool ref);
    matrix(const base_matrix &mat);
    matrix(const base_matrix &mat, node *tracker);
    matrix(const matrix &mat);
    void operator= (const matrix &v);
    void operator= (const base_matrix &v);

    matrix matmul(const base_matrix &mat);
    matrix matmul(const matrix &mat);
    friend matrix linear(const matrix &v);
    friend matrix sigmoid(const matrix &v);
    friend matrix relu(const matrix &v);
    friend matrix logSoftmax(const matrix &v);
};


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

namespace loss{
    class backward_core{
    public:
        node *backward_start;

        backward_core();
        ~backward_core();

        void backward();
        virtual float compute(matrix &predict, const matrix &target) = 0;
    };

    class L2_loss : public backward_core{
    public:
        base_matrix error;
        float compute(matrix &predict, const matrix &target) override;
    };

    class Negative_loglikelihood : public backward_core{
    public:
        float compute(matrix &predict, const matrix &target) override;
    };

}

namespace optim{
    class optim_core{
    public:
        node *back_core;
        float learning_rate;
        optim_core(loss::backward_core &back_core ,float &lr);
        optim_core(node *core ,float &lr);
        ~optim_core();
    };

    class SGD : public optim_core{
    public:
        void step();
        SGD(node *core, float lr);
    };

}

#endif
