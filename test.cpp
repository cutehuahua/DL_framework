#include "loss.h"
#include "nn.h"
#include "dataloader.h"

using namespace std;

int main(){

    //Mnist
    int batch_size = 128;
    nn::Linear layer1(784, 128, nn::Sigmoid, 1);
    nn::Linear layer2(128, 128, nn::Sigmoid, 1);
    nn::Linear layer3(128, 10, nn::LogSoftmax, 1);

    int iteration = 0;
    dataloader data(string("./data"), batch_size);
    for (int epoch = 0; epoch < 5; epoch++){
        while (1){
            matrix batch;
            matrix target;
            if ( !data.get_batch(batch, target) ){
                break; //end of epoch
            }

            matrix x1, x2, x3;
            x1 = layer1.forward(batch);
            x2 = layer2.forward(x1);
            x3 = layer3.forward(x2);

            loss::Negative_loglikelihood loss;

            cout << "epoch : " << epoch << '\t';
            cout << "iteration : " << ++iteration << '\t';
            cout << "loss : " << loss.compute(x3, target) << endl;//flush;
            loss.backward();

            optim::SGD optim(loss.backward_start, 0.0001);
            optim.step();
        }
        cout << endl;
    }


    /*
    //XOR
    float d[8] = {0,0,0,1,1,0,1,1};
    matrix data(4, 2, d, false);

    float t[4] = {0,1,1,0};
    matrix target(4, 1, t, false);
    nn::Linear layer1(2, 2, nn::Sigmoid, 1);
    nn::Linear layer2(2, 2, nn::Sigmoid, 1);
    nn::Linear layer3(2, 1, nn::Sigmoid, 1);

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
    */

    return 0;
}