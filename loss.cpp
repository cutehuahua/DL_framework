#include "loss.h"


loss::backward_core::backward_core(){
    backward_start = new node;
}
loss::backward_core::~backward_core(){
/*
backward_start will be deleted by optim
*/
}
void loss::backward_core::backward(){

    node *tmp = backward_start -> previous;
    base_matrix current_gradient = *backward_start->gradient;
    while (tmp != NULL){
        if (tmp->gradient == NULL){
            if (tmp->bias){
                tmp->weight->height -= 1;
            }
            base_matrix transpose_weight = (*tmp->weight).T();
            base_matrix transpose_data = (*tmp->input_data).T();

            tmp->gradient = new base_matrix( transpose_data._matmul(current_gradient) );
            current_gradient = current_gradient._matmul(transpose_weight);

            if (tmp->bias){
                tmp->weight->height += 1;
            }
        }
        else{
            current_gradient = current_gradient * *tmp->gradient;
        }
        tmp = tmp->previous;
    }
}

float loss::L2_loss::compute(matrix &pred, const matrix &target){
    try{
        if ( (pred.height != target.height) || (pred.width != target.width)){
            throw 1;
        }
    }
    catch (int error){
        std::cerr << "error code : " << error << std::endl;
        std::cerr << "data/target shapes mismatch"  << std::endl; 
        return -1;
    }
    error = (pred - target);

    backward_start -> gradient = new base_matrix(error * 2.0);
    backward_start -> previous = pred.tracker;
    pred.tracker -> next = backward_start;
    
    float l2norm = pow(sum( error*error ), 0.5);
    return l2norm;
}

float loss::Negative_loglikelihood::compute(matrix &pred, const matrix &target){
    try{
        if ( (pred.height != target.height) || target.width != 1 ){
            throw 2;
        }
    }
    catch (int error){
        std::cerr << "error code : " << error << std::endl;
        std::cerr << "# of data mismatch with # of target or target width is not 1"  << std::endl; 
        return -1;
    }

    backward_start -> previous = pred.tracker;
    backward_start -> gradient = new base_matrix(pred.height, pred.width, true);
    pred.tracker -> next = backward_start;

    float loss = 0;
    for (int h = 0; h < pred.height; h++){
            int target_index = static_cast<int>(*(target.value + h) + 0.5 );
            *(backward_start -> gradient -> value + h*pred.width + target_index) = -1;
            loss += pred.at(h, target_index);
    }
    loss /= float(target.height);
    return -loss;
}


optim::optim_core::optim_core(node *core, float &lr){
    learning_rate = lr;
    back_core = core;
}
optim::optim_core::~optim_core(){
    back_core = NULL;
}
optim::SGD::SGD(node *core, float lr) : optim_core(core, lr){}

void optim::SGD::step(){
    node *tmp = back_core;
    node *next;
    static int i = 0;
    while (tmp != NULL){
        if (tmp->weight != NULL && tmp->input_data != NULL){
            *(tmp->weight) -= (*tmp->gradient * learning_rate);
        }
        next = tmp;
        tmp = tmp->previous;
        delete next;
    }
    matrix::tracker = NULL;
}
