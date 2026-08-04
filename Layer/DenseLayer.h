#include "../Matrix/Matrix.h"
#include "../Activations/activations.h"

enum class ActivationType
{
    RELU,
    SOFTMAX
};

class DenseLayer
{
private:
    int inputNo;
    int outputNo;
    Matrix weights;
    Matrix bias;
    Matrix gradWeights;
    Matrix gradBias;
    Matrix lastZ;
    Matrix lastInput;
    ActivationType activation;

public:
    DenseLayer(int inputNo, int outputNo, ActivationType act);

    Matrix forward(const Matrix &inputs);
    Matrix backward(const Matrix &gradient, double learningRate); // gradient = dA, will compute dZ,dW,db and dA for the previous layer
    void zeroGradients();
    void updateWeights(double learningRate, int batchSize);
};