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

Matrix DenseLayer::backward(const Matrix &gradient, bool needInputGrad)
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

    /*
    Matrix dW = dZ * (lastInput.transpose());
    Matrix db = dZ;
    Matrix nextdA = weights.transpose() * dZ;
    We return nextda
    */

    /*
    accumulate gradients
    dW[i][j] = dZ[i] * x[j]
    */

    for (int i = 0; i < outputNo; i++)
    {
        gradBias(i, 0) += dZ(i, 0);
        for (int j = 0; j < inputNo; j++)
        {
            gradWeights(i, j) += dZ(i, 0) * lastInput(j, 0);
        }
    }
    /*
    essentially the loop is doing this but less computationally expensive
    gradWeights += dZ * (lastInput.transpose());
    gradBias += dZ;
    */
    if (!needInputGrad)
        return Matrix(inputNo, 1);

    return weights.transpose() * dZ;
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

void DenseLayer::save(std::ofstream &out) const
{
    weights.saveParameters(out);
    bias.saveParameters(out);
}