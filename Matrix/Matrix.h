#pragma once

#include <iostream>
#include <vector>
#include <fstream>

class Matrix
{
private:
    int rows;
    int cols;
    std::vector<double> data;

public:
    Matrix(int row, int col);

    inline double &operator()(int r, int c)
    {
        return data[cols * r + c];
    }

    inline const double &operator()(int r, int c) const
    {
        return data[cols * r + c];
    }

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

    inline int getRows() const
    {
        return rows;
    }

    inline int getCols() const
    {
        return cols;
    }

    void saveParameters(std::ofstream &out) const;
};