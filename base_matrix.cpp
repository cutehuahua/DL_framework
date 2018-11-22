#include "base_matrix.h"

base_matrix::base_matrix(){
/*
empty base_matrix
reference means whether value pointer is just point to other, not it's own
*/
    height = -1; width = -1;
    value = NULL;
    reference = false;
}

base_matrix::base_matrix(int h, int w, bool zero = false) : height(h), width(w){
/*
new a 1D float with row-major manner to represent a 2D matrix
*/
    if (!zero){
        reference = false;
        value = new float[height * width];
        std::default_random_engine generator;
        std::normal_distribution<float> distribution(0, 1.0);
        for (int i = 0; i < h*w; i++){
            *(value + i) = distribution(generator);
        }
    }
    else{
        reference = false;
        // all elements set to 0
        value = new float[height * width]();
    }
}

base_matrix::~base_matrix(){
/*
free memory of value iff it's not null and it's not others.
*/
    if (value != NULL && reference == false){ delete [] value; value = NULL;}
}

base_matrix::base_matrix(int h, int w, float *val, bool ref) : height(h), width(w){
/*
create a base_matrix with provided float pointer in 2 way : 
1. point to where the pointer point to.
2. deep copy the pointer. TODO : exception control.
*/
    if (ref){
        value = val;
        reference = true;
    }
    else{
        reference = false;
        value = new float[h*w];
        for(int i = 0; i < h*w; i++){
            *(value+i) = *(val+i);
        }
    }
}

base_matrix::base_matrix(const base_matrix &mat){
/*
copy constructor with deep copy manner.
*/
    reference = false;
    height = mat.height; width = mat.width;
    value = new float[height * width];
    for (int i = 0; i < height * width; i++)
        *(value + i) = *(mat.value + i);
}

float base_matrix::at(int h, int w) const{
/*
return value in (h, w) in row-major manner.
*/
    return *(value + width*h + w);
}

/*
following 8 operator only return result base_matrix without change any other.
They are all point-wise operator.
*/
base_matrix base_matrix::operator* (const float &v){
    base_matrix tmp(*this);
    base_matrix_apply(tmp, *(tmp.value + i) *= v;) ;
    return tmp;
}    
base_matrix base_matrix::operator+ (const float &v){
    base_matrix tmp(*this);
    base_matrix_apply(tmp, *(tmp.value + i) += v;) ;
    return tmp;
}    
base_matrix base_matrix::operator- (const float &v){
    base_matrix tmp(*this);
    base_matrix_apply(tmp, *(tmp.value + i) -= v;) ;
    return tmp;
}    
base_matrix base_matrix::operator/ (const float &v){
    base_matrix tmp(*this);
    base_matrix_apply(tmp, *(tmp.value + i) /= v;) ;
    return tmp;
}    
base_matrix base_matrix::operator* (const base_matrix &v){
    base_matrix tmp(*this);
    base_matrix_apply(tmp, *(tmp.value + i) *= *(v.value + i);) ;
    return tmp;
}    
base_matrix base_matrix::operator- (const base_matrix &v){
    base_matrix tmp(*this);
    base_matrix_apply(tmp, *(tmp.value + i) -= *(v.value + i);) ;
    return tmp;
}    
base_matrix base_matrix::operator+ (const base_matrix &v){
    base_matrix tmp(*this);
    base_matrix_apply(tmp, *(tmp.value + i) += *(v.value + i);) ;
    return tmp;
}    
base_matrix base_matrix::operator/ (const base_matrix &v){
    base_matrix tmp(*this);
    base_matrix_apply(tmp, *(tmp.value + i) /= *(v.value + i);) ;
    return tmp;
}  


void base_matrix::operator-= (const base_matrix &v){
/*
point-wise operation between two base_matrix without any deep copy. 
*/
    try{
        if (height != v.height || width != v.width){
            throw error_code::shape_mismatch;
        }
        for(int i = 0 ; i < height*width; i++ ){
            *(value + i) -= *(v.value + i);
        }
    }
    catch(int error){
        std::cerr << "error code : " << error << std::endl;
        std::cerr << "shape mistach, cannot do -= " << std::endl;
        std::exit;
    }
}
void base_matrix::operator-= (const float &v){
/*
point-wise operation between two base_matrix without any deep copy. 
*/
    for(int i = 0 ; i < height*width; i++ ){
        *(value + i) -= v;
    }
}
void base_matrix::operator= (const base_matrix &v){
/*
deep copy &v
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

base_matrix base_matrix::_matmul(const base_matrix &mat){
/*
matrix multiplication
*/
    try{
        if (width != mat.height){
            throw error_code::shape_mismatch;
        }
        //don't waste time on random initial 
        base_matrix tmp(height, mat.width, true);
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
        std::exit;
    }
}
base_matrix base_matrix::T(){
/*
matrix transpose, return a new base_matrix.  
*/
    base_matrix tmp(width, height, true);
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
float base_matrix::max(){
/*
return max value

TODO : return location as well.

*/    
    float max = -1e9;
    base_matrix tmp(height, width, value, true);
    base_matrix_apply( tmp, if(*(value+i) > max) max = *(value+i););
    return max;
}
void base_matrix::row_assign(const base_matrix &v, int row){
/*
assign a row to this base_matrix
*/    
    try{
        if (v.width != width){
            throw error_code::shape_mismatch;
        }
        for (int w = 0; w < width; w++){
            *(value + row*width + w) = v.at(0, w);
        }
    }
    catch (int error){
        std::cerr << "cannot assign this row, width mismatch" << std::endl;
        std::exit;
    }
}
base_matrix pow(const base_matrix &v, float p){
    base_matrix tmp(v);
    base_matrix_apply(tmp, *(tmp.value + i) = pow(*(tmp.value+i), p););
    return tmp;
}
base_matrix exp(const base_matrix &v){
    base_matrix tmp(v);
    base_matrix_apply(tmp, *(tmp.value + i) = exp( *(tmp.value+i) ););
    return tmp;
}
float sum(const base_matrix &v){
    float sum(0);
    base_matrix_apply(v, sum += *(v.value+i); );
    return sum;
}
float row_sum(const base_matrix &mat, int row){
    float sum(0);
    for (int w = 0; w < mat.width; w++){
        sum += *(mat.value + row*mat.width + w);
    }
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

base_matrix logsoftmax_one_row(const base_matrix &v){
    try{
        if ( v.height != 1){
            throw base_matrix::error_code::shape_mismatch;
        }
        base_matrix tmp(v);
        float row_max = tmp.max();
        tmp -= row_max;
        base_matrix exp_tmp = exp(tmp);
        float row_sum_log = log(row_sum(exp_tmp, 0));
        tmp -= row_sum_log;
        return tmp;
    }
    catch(int error){
        std::cerr << "need to be one row only" << std::endl;
        std::exit;
    }
}
base_matrix logSoftmax(const base_matrix &v){
/*
deep copy each row to do logsoftmax_one_row and assign back to output base_matrix
*/
    base_matrix out(v.height, v.width);
    for (int h = 0; h < v.height; h++){
        base_matrix tmp(1, v.width, (v.value + h*v.width), false);
        tmp = logsoftmax_one_row(tmp);
        out.row_assign(tmp, h);
    }
    return out;
}

base_matrix cat(const base_matrix &a, const base_matrix &b, int dim){
/*
dim 0 : height
dim 1 : width

return a deep copy of cat(a, b)
*/
    try{
        if (dim == 0){
            if(a.width != b.width)
                throw base_matrix::error_code::shape_mismatch;
            base_matrix out(a.height + b.height, a.width, true);
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
                throw base_matrix::error_code::shape_mismatch;
            base_matrix out(a.height , a.width + b.width, true);
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
        std::exit;
    }

}
