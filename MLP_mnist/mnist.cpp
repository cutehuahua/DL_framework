#include "core.h"
#include "dataloader.h"

using namespace std;


int main(){

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

            cout << '\r' << "epoch : " << epoch << '\t';
            cout << "iteration : " << ++iteration << '\t';
            cout << "loss : " << loss.compute(x3, target) << flush;
            loss.backward();

            optim::SGD optim(loss.backward_start, 0.0001);
            optim.step();
        }
        cout << endl;
    }
    return 0;
}

