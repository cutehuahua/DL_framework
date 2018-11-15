#include "xor.h"

//base matrix
base_matrix::base_matrix(){
    height = -1; width = -1;
    value = NULL;
    reference = false;
}
base_matrix::base_matrix(int h, int w) : height(h), width(w){
    reference = false;
    value = new float[height * width];
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0, 1.0);
    for (int i = 0; i < h*w; i++){
        *(value + i) = distribution(generator);
    }
}
base_matrix::~base_matrix(){
    if (value != NULL && reference == false){ delete [] value; value = NULL;}
}
base_matrix::base_matrix(int h, int w, float *val, bool ref) : height(h), width(w){
    if (ref){
        value = val;
        reference = true;
    }
    else{
        reference = false;
        value = new float[h*w];
        for(int i = 0; i < h*w; i++)
            *(value+i) = *(val+i);
    }
}
//deep-copy
base_matrix::base_matrix(const base_matrix &mat){
    reference = false;
    height = mat.height; width = mat.width;
    value = new float[height * width];
    for (int i = 0; i < height * width; i++)
        *(value + i) = *(mat.value + i);
}

base_matrix::base_matrix(const base_matrix *mat){
    reference = false;
    height = mat->height; width = mat->width;
    value = new float[height * width];
    for (int i = 0; i < height * width; i++)
        *(value + i) = *(mat->value + i);
}
//row-major
float base_matrix::at(int h, int w) const{
    return *(value + ((width*h) + w));
}
base_matrix base_matrix::operator* (const float &v){
    base_matrix tmp(this);
    base_matrix_apply(tmp, *(tmp.value + i) *= v;) ;
    return tmp;
}    
base_matrix base_matrix::operator+ (const float &v){
    base_matrix tmp(this);
    base_matrix_apply(tmp, *(tmp.value + i) += v;) ;
    return tmp;
}    
base_matrix base_matrix::operator- (const float &v){
    base_matrix tmp(this);
    base_matrix_apply(tmp, *(tmp.value + i) -= v;) ;
    return tmp;
}    
base_matrix base_matrix::operator/ (const float &v){
    base_matrix tmp(this);
    base_matrix_apply(tmp, *(tmp.value + i) /= v;) ;
    return tmp;
}    
base_matrix base_matrix::operator* (const base_matrix &v){
    base_matrix tmp(this);
    base_matrix_apply(tmp, *(tmp.value + i) *= *(v.value + i);) ;
    return tmp;
}    
base_matrix base_matrix::operator- (const base_matrix &v){
    base_matrix tmp(this);
    base_matrix_apply(tmp, *(tmp.value + i) -= *(v.value + i);) ;
    return tmp;
}    
base_matrix base_matrix::operator+ (const base_matrix &v){
    base_matrix tmp(this);
    base_matrix_apply(tmp, *(tmp.value + i) += *(v.value + i);) ;
    return tmp;
}    
base_matrix base_matrix::operator/ (const base_matrix &v){
    base_matrix tmp(this);
    base_matrix_apply(tmp, *(tmp.value + i) /= *(v.value + i);) ;
    return tmp;
}    

void base_matrix::operator= (const base_matrix &v){

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

base_matrix base_matrix::_matmul(const base_matrix &mat){
    try{
        if (width != mat.height){
            throw -1;
        }

        base_matrix tmp(height, mat.width);
        for (int h = 0; h < height; h++){
            for (int w = 0; w < mat.width; w++){

                float val(0);
                for (int i = 0; i < width; i++){
                    val += ( at(h, i) * mat.at(i, w) );
                }
                *(tmp.value + h * mat.width + w) = val;
            }
        }
        return tmp;
    }
    catch (int i){
        std::cerr << "error code : " << i << std::endl;
        std::cerr << "wrong dimension, got "  << std::endl; 
        std::cerr << height << '\t' << width << std::endl;
        std::cerr << mat.height << '\t' << mat.width << std::endl;
    }
}
base_matrix base_matrix::T(){
    base_matrix tmp(width, height);
    for (int h = 0; h < height; h++){
        for (int w = 0; w < width; w++){
            *(tmp.value + w * height + h) = at(h, w);
        }
    }
    return tmp;
}
base_matrix base_matrix::transpose(){
    return T();
}

//friends
base_matrix pow(const base_matrix &v, float p){
    base_matrix tmp(v);
    base_matrix_apply(tmp, *(tmp.value + i) = pow(*(tmp.value+i), p););
    return tmp;
}
float sum(const base_matrix &v){
    float sum(0);
    base_matrix_apply(v, sum += *(v.value+i); );
    return sum;
}
std::ostream& operator<< (std::ostream& out, const base_matrix &mat){
    std::cout << "shape : " << mat.height << '\t' << mat.width << std::endl;
    for (int h = 0; h < mat.height; h++){
        for (int w = 0; w < mat.width; w++)
            std::cout << *(mat.value + h*mat.width + w)  << '\t';
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
base_matrix sigmoid(const base_matrix &v){
    base_matrix tmp(v);
    base_matrix_apply(tmp, *(tmp.value + i) = 1.0 / (1.0 + exp(-*(tmp.value + i))); );
    return tmp;
}
base_matrix relu(const base_matrix &v){
    base_matrix tmp(v);
    base_matrix_apply(tmp, *(tmp.value + i) = (*(tmp.value + i) > 0) ? *(tmp.value + i) : 0;);
    return tmp;
}

matrix sigmoid(const matrix &mat){
    node *tmp_tracker = new node;
    base_matrix tmp(sigmoid(base_matrix(mat)));
    if (mat.tracker != NULL){ 
        node *t = new node;
        *tmp_tracker = *mat.tracker;

        t -> previous = tmp_tracker;
        tmp_tracker -> next = t;
        tmp_tracker = t;
    }
    tmp_tracker -> gradient = new base_matrix(tmp.height, tmp.width);

    for (int i = 0; i < tmp.height*tmp.width; i++){
        float sigmoid_out = *(tmp.value + i);
        *(tmp_tracker -> gradient -> value + i) = sigmoid_out * (1.0 - sigmoid_out); 
    }
    return matrix(tmp, tmp_tracker);
}
matrix relu(const matrix &mat){
    node *tmp_tracker = new node;
    base_matrix tmp(relu(base_matrix(mat)));

    if (mat.tracker != NULL){ 
        node *t = new node;
        *(tmp_tracker) = *(mat.tracker);

        t -> previous = tmp_tracker;
        tmp_tracker -> next = t;
        tmp_tracker = t;
    }
    tmp_tracker -> gradient = new base_matrix(tmp.height, tmp.width);

    for (int i = 0; i < tmp.height*tmp.width; i++){
        float relu_out = *(tmp.value + i);
        *(tmp_tracker -> gradient -> value + i) = (relu_out > 0.0)? 1.0 : 0.0 ; 
    }
    return matrix(tmp, tmp_tracker);
}
base_matrix cat(const base_matrix &a, const base_matrix &b, int dim){
    try{
        if (dim == 0){
            if(a.width != b.width)
                throw -1;
            matrix out(a.height + b.height, a.width);
            for (int h = 0; h < out.height; h++){
                for (int w = 0; w < out.width; w++){
                    if (h < a.height)
                        *(out.value + (h*out.width) + w) = a.at(h, w); 
                    else
                        *(out.value + (h*out.width) + w) = b.at(h - a.height, w);
                }
            } 
            return out;
        }
        else if(dim == 1){
            if (a.height != b.height)
                throw -1;
            matrix out(a.height , a.width + b.width);
            for (int h = 0; h < out.height; h++){
                for (int w = 0; w < out.width; w++){
                    if (w < a.width)
                        *(out.value + (h*out.width) + w) = a.at(h, w); 
                    else
                        *(out.value + (h*out.width) + w) = b.at(h, w - a.width);
                }
            } 
            return out;
        }
        else if(dim > 1)
            throw -1;
    }
    catch(int error){
        std::cerr << "dimension size cannot be matched" << std::endl; 
        std::exit(-1);
    }

}

//matrix
matrix::matrix() : base_matrix(){
    tracker = NULL;
}
matrix::matrix(int h, int w) : base_matrix(h, w){
    tracker = NULL;
}
matrix::matrix(int h, int w, float *val, bool ref) : base_matrix(h, w, val, ref){
    tracker = NULL;
}
matrix::matrix(const base_matrix &mat) : base_matrix(mat){
    tracker = NULL;
}
matrix::matrix(const base_matrix &mat, node* t): base_matrix(mat){
    tracker = NULL;
    if(t != NULL){
        tracker = new node;
        *tracker = *t;
    }
}
matrix::matrix(const matrix &mat) : base_matrix(mat){
    if (tracker != NULL){
        tracker = NULL;
    }
    if (mat.tracker != NULL){
        tracker = new node;
        *(tracker) = *(mat.tracker);
    }
}
matrix::~matrix(){
    if (tracker != NULL){
        delete tracker;
        tracker = NULL;
    }
}
void matrix::operator= (const base_matrix &v){
    height = v.height; width = v.width;
    if (value != NULL){
        delete [] value;
        value = NULL;
    }
    value = new float[height * width];
    for (int i = 0; i < height*width; i++){
        *(value + i) = *(v.value + i);
    }
    tracker = NULL;
}
void matrix::operator= (const matrix &v){
    height = v.height; width = v.width;
    if (value != NULL){
        delete [] value;
        value = NULL;
    }
    value = new float[height * width];
    for (int i = 0; i < height*width; i++){
        *(value + i) = *(v.value + i);
    }
    if (tracker == NULL && v.tracker != NULL){
        tracker = new node;
    }
    if (v.tracker != NULL){
        *(tracker) = *(v.tracker);
    }
}
matrix matrix::matmul(const matrix &mat){
    node *tmp_tracker = new node;
    if (tracker != NULL){ 
        *(tmp_tracker) = *(tracker);
        node *t = new node;
        t -> previous = tmp_tracker;
        tmp_tracker -> next = t;
        tmp_tracker = t;
    }
    tmp_tracker -> input_data = new base_matrix(height, width, value, false);
    tmp_tracker -> weight =  new base_matrix(mat.height, mat.width, mat.value, true);

    return matrix(_matmul(mat), tmp_tracker);
}

//node
node::node(): previous(NULL), next(NULL), input_data(NULL), 
              weight(NULL), gradient(NULL), bias(false){}
node::~node(){
    if (next != NULL){ next = NULL; }
    if (previous != NULL){ previous = NULL;}
    if (weight != NULL){ weight = NULL;}
    if (input_data != NULL){ input_data = NULL;}
    if (gradient != NULL){ delete gradient; gradient = NULL;}
}
void node::operator=(const node &v){
    bias = v.bias;
    next = (v.next == NULL)? NULL : v.next;
    previous = (v.previous == NULL)? NULL : v.previous;
    input_data = (v.input_data == NULL)? NULL : v.input_data;
    weight = (v.weight == NULL)? NULL : v.weight;

    if (gradient != NULL){
        delete gradient;
        gradient = NULL;
    }
    if (v.gradient != NULL){
        gradient = new base_matrix;
        *(gradient) = *(v.gradient);
    }
}

loss::L2_loss::L2_loss(){
    backward_start = new node;
}
loss::L2_loss::~L2_loss(){
    delete backward_start;
}

float loss::L2_loss::compute(matrix &pred, const matrix &target){
    try{
        if ( (pred.height != target.height) || (pred.width != target.width)){
            throw 1;
        }
    }
    catch (int error){
        std::cerr << "error code : " << error << std::endl;
        std::cerr << "wrong shape"  << std::endl; 
        return -1;
    }
    error = (pred - target);

    backward_start -> gradient = new base_matrix(error * 2.0);
    backward_start -> previous = pred.tracker;
    
    float l2norm = pow(sum( error*error ), 0.5);
    return l2norm;
}

void loss::L2_loss::backward(){

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
        }
        else{
            current_gradient = current_gradient * *tmp->gradient;
        }
        tmp = tmp->previous;
    }
}

optim::optim_core::optim_core(node *core, float &lr){
    learning_rate = lr;
    back_core = core;
}
optim::SGD::SGD(node *core, float lr) : optim_core(core, lr){}

void optim::SGD::step(){
    node *tmp = back_core;
    while (tmp != NULL){
        if (tmp->weight != NULL && tmp->input_data != NULL){
            *(tmp->weight) = (*tmp->weight - (*tmp->gradient * learning_rate));
        }
        tmp = tmp->previous;
    }
}

nn::linear::linear(int in_dim, int out_dim, int nonlinear, bool if_bias){
    if (if_bias){
        bias = if_bias;
        weight = new matrix(in_dim + 1, out_dim);
    }
    else{
        bias = if_bias;
        weight = new matrix(in_dim, out_dim);
    }

    switch (nonlinear){
        case 0:
            activation = relu;
            break; 
        case 1:
            activation = sigmoid;
            break;
        default:
            activation = relu;
    }
}
nn::linear::~linear(){
    delete weight;
}

matrix nn::linear::forward(const matrix &data){
    if (bias){
        float *bptr = new float[data.height];
        for(int i = 0; i < data.height; i++)
            *(bptr + i) = 1;
        matrix b(data.height, 1, bptr, false);
        delete [] bptr;

        matrix data_with_bias = cat(data, b, 1);
        if (data.tracker != NULL){
            data_with_bias.tracker = new node;
            *data_with_bias.tracker = *data.tracker;
        }
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

int main(){

    float d[8] = {0,0,0,1,1,0,1,1};
    matrix data(4, 2, d, false);

    float t[4] = {0,1,1,0};
    matrix target(4, 1, t, false);
    nn::linear layer1(2, 2, nn::Sigmoid, true);
    nn::linear layer2(2, 2, nn::Sigmoid, true);
    nn::linear layer3(2, 1, nn::Sigmoid, true);

    for(int iter = 0; iter < 50000; iter++){
        matrix x1, x2, x3;
        x1 = layer1.forward(data);
        x2 = layer2.forward(x1);
        x3 = layer3.forward(x2);

        loss::L2_loss l2;
        std::cout << l2.compute(x3, target) << std::endl;
        std::cout << x3;
        l2.backward();

        optim::SGD optim(l2.backward_start, 0.1);
        optim.step();
    }

    return 0;
}
