#include "DenseLayer.h"
#include "../Matrix/Matrix.h"
#include "../Activations/activations.h"
#include <cmath>

using namespace std;

DenseLayer::DenseLayer(int input, int output, ActivationType act) : inputNo(input),
                                                                    outputNo(output),
                                                                    weights(output, input),
                                                                    bias(output, 1),
                                                                    lastZ(output, 1),
                                                                    lastInput(input, 1),
                                                                    activation(act)
{
    weights.randomize(0.0, sqrt(2.0 / inputNo)); // He initialisation
    bias.fill(0.0);
}

Matrix DenseLayer::forward(const Matrix &m)
{
    Matrix ans = weights * m + bias;
    lastInput = m;
    lastZ = ans;
    return ans;
}

Matrix DenseLayer::backward(const Matrix &gradient, double learningRate)
{
    Matrix dZ(outputNo, 1);

    if (activation == ActivationType::RELU)
    {
        dZ = gradient.hadamard(reluDerivative(lastZ));
    }
    else
    {
        dZ = gradient;
    }

    Matrix dW = dZ * (lastInput.transpose());
    Matrix db = dZ;
    Matrix nextdA = weights.transpose() * dZ;

    weights -= dW * learningRate;
    bias -= db * learningRate;

    return nextdA;
}