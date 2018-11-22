#ifndef LOSS_H
#define LOSS_H
#include "matrix.h"
#include "node.h"

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
