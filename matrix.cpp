#include "matrix.h"

/*
gradient tracker is static variable in this class
*/
node* matrix::tracker = NULL;

matrix::matrix() : base_matrix(){}
matrix::matrix(int h, int w) : base_matrix(h, w, true){}
matrix::matrix(int h, int w, float *val, bool ref) : base_matrix(h, w, val, ref){}
matrix::matrix(const base_matrix &mat) : base_matrix(mat){}
matrix::~matrix(){}

void matrix::operator= (const matrix &v){
/*
deep copy of v
*/
    height = v.height; width = v.width;
    if (value != NULL){
        delete [] value;
        value = NULL;
    }
    value = new float[height * width];
    for (int i = 0; i < height*width; i++){
        *(value + i) = *(v.value + i);
    }
}
void matrix::operator= (const base_matrix &v){
/*
deep copy of v
*/
    height = v.height; width = v.width;
    if (value != NULL){
        delete [] value;
        value = NULL;
    }
    value = new float[height * width];
    for (int i = 0; i < height*width; i++){
        *(value + i) = *(v.value + i);
    }
}


matrix matrix::matmul(const base_matrix &mat){

    node *tmp_tracker = new node;
    tmp_tracker -> input_data = new base_matrix(height, width, value, false);
    tmp_tracker -> weight =  new base_matrix(mat.height, mat.width, mat.value, true);

    node_append(tracker, tmp_tracker);
    return matrix(_matmul(mat));
}

matrix logSoftmax(const matrix &mat){
    base_matrix tmp(logSoftmax(base_matrix(mat)));

    node *tmp_tracker = new node;
    tmp_tracker -> gradient = new base_matrix(tmp.height, tmp.width, true);
    for (int h = 0; h < tmp.height; h++){
        for (int w = 0; w < tmp.width; w++){
            float logsoftmax_out = tmp.at(h, w);
            *(tmp_tracker -> gradient -> value + h*tmp.width + w) = 1.0 - logsoftmax_out * tmp.width; 
        }
    }
    node_append(matrix::tracker, tmp_tracker);
    return matrix(tmp);
}
matrix sigmoid(const matrix &mat){
    base_matrix tmp(sigmoid(base_matrix(mat)));

    node *tmp_tracker = new node;
    tmp_tracker -> gradient = new base_matrix(tmp.height, tmp.width, true);
    for (int i = 0; i < tmp.height*tmp.width; i++){
        float sigmoid_out = *(tmp.value + i);
        *(tmp_tracker -> gradient -> value + i) = sigmoid_out * (1.0 - sigmoid_out); 
    }
    node_append(matrix::tracker, tmp_tracker);
    return matrix(tmp);
}

matrix relu(const matrix &mat){
    base_matrix tmp(relu(base_matrix(mat)));

    node *tmp_tracker = new node;
    tmp_tracker -> gradient = new base_matrix(tmp.height, tmp.width, true);
    for (int i = 0; i < tmp.height*tmp.width; i++){
        float relu_out = *(tmp.value + i);
        *(tmp_tracker -> gradient -> value + i) = (relu_out > 0.0)? 1.0 : 0.0 ; 
    }
    node_append(matrix::tracker, tmp_tracker);
    return matrix(tmp);
}
matrix linear(const matrix &mat){
    base_matrix tmp(mat);

    node *tmp_tracker = new node;
    tmp_tracker -> gradient = new base_matrix(tmp.height, tmp.width, true);
    for (int i = 0; i < tmp.height*tmp.width; i++){
        *(tmp_tracker -> gradient -> value + i) = 1.0; 
    }
    node_append(matrix::tracker, tmp_tracker);
    return matrix(tmp);
}
