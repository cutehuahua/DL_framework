#include "node.h"

//node
node::node(): previous(NULL), next(NULL), input_data(NULL), 
              weight(NULL), gradient(NULL), bias(false), nonlinear_layer(false) {}
node::~node(){
/*
all information except gradient should be reference
*/
    if (next != NULL){ next = NULL; }
    if (previous != NULL){ previous = NULL;}
    if (weight != NULL){ weight = NULL;}
    if (input_data != NULL){ input_data = NULL;}
    if (gradient != NULL){ delete gradient; gradient = NULL;}
}


void node_append(node *&a, node *b){
/*
append b after a
*/
    if (a == NULL){
        if (b == NULL){
            a = NULL;
        }
        else{
            a = b;
        }
    }
    else{
        if (b != NULL){
            a -> next = b;
            b -> previous = a;
            a = b;
        }
    }
}