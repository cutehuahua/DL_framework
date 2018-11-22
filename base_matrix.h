#ifndef BASE_M_H
#define BASE_M_H
#include <iostream>
#include <math.h>
#include <random>
#include <cstdlib>

#define base_matrix_apply(mat, op_code)                      \
    for (int i = 0; i < mat.height * mat.width; i++){ \
        op_code}

class base_matrix{
public:
    enum error_code{shape_mismatch};
    float *value;
    int height, width;
    bool reference;

    ~base_matrix();
    base_matrix();
    base_matrix(int h, int w, bool fill_zero);
    base_matrix(int h, int w, float *value, bool ref);
    base_matrix(const base_matrix &mat);
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
    void operator-= (const float &v);
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


#endif
