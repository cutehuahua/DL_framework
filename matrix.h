#ifndef MATRIX_H
#define MATRIX_H
#include <iostream>
#include "node.h"
#include "base_matrix.h"


class matrix : public base_matrix{
public:
    static node *tracker;

    matrix();
    ~matrix();
    matrix(int h, int w);
    matrix(int h, int w, float *value, bool ref);
    matrix(const base_matrix &mat);
    void operator= (const matrix &v);
    void operator= (const base_matrix &v);

    matrix matmul(const base_matrix &mat);
    friend matrix linear(const matrix &v);
    friend matrix sigmoid(const matrix &v);
    friend matrix relu(const matrix &v);
    friend matrix logSoftmax(const matrix &v);
};

#endif
