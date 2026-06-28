#pragma once

#include <iostream>
#include <vector>

class Matrix
{
private:
    int rows;
    int cols;
    std::vector<double> data;

public:
    Matrix(int row, int col);

    double &operator()(int row, int col);
    const double &operator()(int r, int c) const;

    Matrix operator*(const Matrix &other) const;
    Matrix operator*(double scalar) const;
    Matrix operator+(const Matrix &other) const;
    Matrix operator-(const Matrix &other) const;
    Matrix &operator+=(const Matrix &other);
    Matrix &operator-=(const Matrix &other);

    Matrix transpose() const;
    Matrix hadamard(const Matrix &other) const;

    void fill(double value);
    void randomize(double mean, double stddev);
    void print() const;

    int getRows() const;
    int getCols() const;
};