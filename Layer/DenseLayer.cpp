#include "DenseLayer.h"
#include "../Matrix/Matrix.h"
#include "../Activations/activations.h"
#include <cmath>

using namespace std;

DenseLayer::DenseLayer(int input, int output, ActivationType act) : inputNo(input),
                                                                    outputNo(output),
                                                                    weights(output, input),
                                                                    bias(output, 1),
                                                                    gradWeights(output, input),
                                                                    gradBias(output, 1),
                                                                    lastZ(output, 1),
                                                                    lastInput(input, 1),
                                                                    activation(act)
{
    weights.randomize(0.0, sqrt(2.0 / inputNo)); // He initialisation
    bias.fill(0.0);

    gradWeights.fill(0);
    gradBias.fill(0);
}

Matrix DenseLayer::forward(const Matrix &m)
{
    if (m.getRows() != inputNo || m.getCols() != 1)
    {
        throw invalid_argument("Forward input has incorrect dimensions.");
    }
    Matrix ans = weights * m + bias;
    lastInput = m;
    lastZ = ans;
    return ans;
}

Matrix DenseLayer::backward(const Matrix &gradient, double learningRate)
{
    if (gradient.getRows() != outputNo || gradient.getCols() != 1)
    {
        throw invalid_argument("Backward: Gradient has incorrect dimensions.");
    }

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

    // accumulate gradients
    gradWeights += dW;
    gradBias += db;

    return nextdA;
}

void DenseLayer::updateWeights(double learningRate, int batchSize)
{
    weights -= gradWeights * (learningRate / batchSize);
    bias -= gradBias * (learningRate / batchSize);
}

void DenseLayer::zeroGradients()
{
    gradWeights.fill(0.0);
    gradBias.fill(0.0);
}