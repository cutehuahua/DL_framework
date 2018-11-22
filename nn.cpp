#include "nn.h"

extern matrix linear(const matrix &v);
extern matrix sigmoid(const matrix &v);
extern matrix relu(const matrix &v);
extern matrix logSoftmax(const matrix &v);

nn::Linear::Linear(int in_dim, int out_dim, int nonlinear, bool if_bias){
    if (if_bias){
        bias = if_bias;
        weight = new base_matrix(in_dim + 1, out_dim, false);
    }
    else{
        bias = if_bias;
        weight = new base_matrix(in_dim, out_dim, false);
    }

    switch (nonlinear){
        case 0:
            activation = relu;
            break; 
        case 1:
            activation = sigmoid;
            break;
        case 2:
            activation = logSoftmax;
            break;
        case 3:
            activation = linear;
            break;
        default:
            activation = linear;
    }
}
nn::Linear::~Linear(){
    delete weight;
}

matrix nn::Linear::forward(const matrix &data){
    if (bias){

        //append bias term
        float *bptr = new float[data.height];
        for(int i = 0; i < data.height; i++)
            *(bptr + i) = 1;
        matrix b(data.height, 1, bptr, true);
        matrix data_with_bias = cat(data, b, 1);
        delete [] bptr;

        matrix hidden = data_with_bias.matmul(*weight);
        hidden.tracker->bias = true;
        matrix active = activation(hidden);

        return active;
    }
    else{
        matrix tmp(data.height, data.width, data.value, true);
        matrix hidden = tmp.matmul(*weight);

        hidden.tracker->bias = false;
        matrix active = activation(hidden);

        return active;
    }
}

