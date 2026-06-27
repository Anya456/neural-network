#pragma once

#include <iostream>
#include <vector>
#include "../Matrix/Matrix.h"

Matrix relu(const Matrix &m); // reLu(a) = max(a,0)
Matrix reluDerivative(const Matrix &m);

Matrix softmax(const Matrix &m);
int argmax(const Matrix &m);
double crossEntropy(const Matrix &predicted, int corrIndex);
