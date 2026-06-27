#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "activations.h"

using namespace std;

Matrix relu(const Matrix &m)
{
    int r = m.getRows();
    int c = m.getCols();
    Matrix ans(r, c);
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            ans(i, j) = max(m(i, j), 0.0);
        }
    }
    return ans;
}

Matrix reluDerivative(const Matrix &m)
{
    int r = m.getRows();
    int c = m.getCols();
    Matrix ans(r, c);

    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            if (m(i, j) <= 0)
                ans(i, j) = 0;
            else
                ans(i, j) = 1;
        }
    }
    return ans;
}

Matrix softmax(const Matrix &m)
{
    int r = m.getRows();
    int c = m.getCols();
    if (c != 1)
        throw invalid_argument("softmax excepts column vector");

    double maxEl = m(0, 0);
    Matrix ans(r, 1);

    for (int i = 0; i < r; i++)
    {
        maxEl = max(maxEl, m(i, 0));
    }

    double sum = 0.0;

    for (int i = 0; i < r; i++)
    {
        ans(i, 0) = exp(m(i, 0) - maxEl);
        sum += ans(i, 0);
    }

    for (int i = 0; i < r; i++)
    {
        ans(i, 0) = ans(i, 0) / sum;
    }
    return ans;
}

int argmax(const Matrix &m)
{
    if (m.getCols() != 1)
        throw invalid_argument("argmax expects column vector.");
    int r = m.getRows();
    int maxIndex = 0;
    double maxEl = m(0, 0);

    for (int i = 0; i < r; i++)
    {
        if (m(i, 0) > maxEl)
        {
            maxEl = m(i, 0);
            maxIndex = i;
        }
    }

    return maxIndex;
}

double crossEntropy(const Matrix &predicted, int index)
{
    if (index < 0 || index >= predicted.getRows())
        throw invalid_argument("Index out of range");
    return -log(predicted(index, 0) + 1e-8);
}
