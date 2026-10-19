#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <fstream>

#include "../Matrix/Matrix.h"
#include "../Activations/activations.h"
#include "../DataLoader/dataLoader.h"
#include "../Layer/DenseLayer.h"

using namespace std;

void printImage(const Matrix &img)
{
    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            cout << (img(i * 28 + j, 0) < 0.5 ? '.' : '#');
        }
        cout << endl;
    }
}

void exportSample(const string &path, const Matrix &img, int label)
{
    ofstream out(path);
    out << "{\"label\": " << label << ", \"pixels\": [";
    for (int i = 0; i < 784; i++)
    {
        out << (i ? "," : "") << img(i, 0);
    }
    out << "]}" << endl;
}

int main()
{
    // load data
    vector<Matrix> trainImages = loadImages("data/emnist-balanced-train-images-idx3-ubyte");
    vector<int> trainLabels = loadLabels("data/emnist-balanced-train-labels-idx1-ubyte");
    vector<Matrix> testImages = loadImages("data/emnist-balanced-test-images-idx3-ubyte");
    vector<int> testLabels = loadLabels("data/emnist-balanced-test-labels-idx1-ubyte");

    bool previewOnly = false;

    if (previewOnly)
    {
        const string LABELS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt";
        for (int i = 0; i < 4; i++)
        {
            cout << "Label: " << trainLabels[i] << ": " << LABELS[trainLabels[i]] << endl;
            printImage(trainImages[i]);
            cout << endl;
        }
        return 0;
    }

    // make samples
    vector<Sample> trainData = makeSamples(trainImages, trainLabels);
    vector<Sample> testData = makeSamples(testImages, testLabels);

    std::cout << "Training dataset loaded. Size : " << trainData.size() << endl;
    std::cout << "Testing dataset loaded. Size : " << testData.size() << endl;
    exportSample("sample.json", testImages[0], testLabels[0]);

    // create layers
    DenseLayer layer1(784, 256, ActivationType::RELU);
    DenseLayer layer2(256, 128, ActivationType::RELU);
    DenseLayer layer3(128, 47, ActivationType::SOFTMAX);

    std::cout << "Dense layers made" << endl;

    // hyperparameters
    double learningRate = 0.05;
    int epochs = 15;
    int classNo = 47;
    int batchSize = 32;
    int n = trainData.size();

    // random number generators
    random_device rd;
    mt19937 gen(rd());

    for (int i = 0; i < epochs; i++)
    {
        cout << "Epoch " << i << " started" << endl;
        auto startTime = chrono::steady_clock::now();

        shuffle(trainData.begin(), trainData.end(), gen);

        double loss = 0.0;
        int correct = 0;

        for (int start = 0; start < n; start += batchSize)
        {
            int end = min(start + batchSize, n);

            layer1.zeroGradients();
            layer2.zeroGradients();
            layer3.zeroGradients();

            for (int j = start; j < end; j++)
            {

                const Matrix &input = trainData[j].image;
                int label = trainData[j].label;

                /* forward pass
                Matrix z1 = layer1.forward(input);
                Matrix a1 = relu(z1);
                Matrix z2 = layer2.forward(a1);
                Matrix a2 = relu(z2);
                Matrix z3 = layer3.forward(a2);
                Matrix a3 = softmax(z3);
                */

                Matrix a1 = relu(layer1.forward(input));
                Matrix a2 = relu(layer2.forward(a1));
                Matrix a3 = softmax(layer3.forward(a2));

                // compute loss
                loss += crossEntropy(a3, label);

                // check accuracy
                int predicted = argmax(a3);
                if (predicted == label)
                    correct++;

                // build yTrue
                Matrix yTrue(classNo, 1);
                yTrue.fill(0);
                yTrue(label, 0) = 1;

                // back propagation
                Matrix gradient = a3 - yTrue;         // dZ3
                gradient = layer3.backward(gradient); // gives dA2
                gradient = layer2.backward(gradient); // gives dA1
                gradient = layer1.backward(gradient, false);
            }
            layer3.updateWeights(learningRate, end - start);
            layer2.updateWeights(learningRate, end - start);
            layer1.updateWeights(learningRate, end - start);
        }

        auto endTime = chrono::steady_clock::now();
        double seconds = chrono::duration<double>(endTime - startTime).count();

        cout << "Epoch " << i + 1
             << " | loss: " << loss / n
             << " | accuracy: " << 100.0 * correct / n << "%"
             << " | time: " << seconds << "s" << endl;
    }

    ofstream out("model.bin", ios::binary);
    layer1.save(out);
    layer2.save(out);
    layer3.save(out);
    out.close();
    cout << "Saved model.bin" << endl;

    Matrix p = softmax(layer3.forward(relu(layer2.forward(relu(layer1.forward(testImages[0]))))));
    cout << "Reference output for sample.json (true label " << testLabels[0] << "):" << endl;
    p.print();
    cout << "Predicted class: " << argmax(p) << endl;

    int testCorrect = 0;
    for (int i = 0; i < (int)testData.size(); i++)
    {
        const Matrix &input = testData[i].image;
        int label = testData[i].label;

        Matrix a1 = relu(layer1.forward(input));
        Matrix a2 = relu(layer2.forward(a1));
        Matrix a3 = softmax(layer3.forward(a2));

        if (argmax(a3) == label)
            testCorrect++;
    }
    double testAccuracy = 100.0 * testCorrect / testData.size();
    std::cout << "Test Accuracy: " << testAccuracy << "%" << endl;
}