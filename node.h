#ifndef NODE_H
#define NODE_H
#include "node.h"
#include "base_matrix.h"


class node{
public:
    node *next, *previous;
    base_matrix *input_data, *weight, *gradient;
    bool bias;
    bool nonlinear_layer;

    node();
    ~node();
    friend void node_append(node *&a, node *v);
};


#endif
