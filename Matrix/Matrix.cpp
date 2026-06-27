#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>
#include <algorithm>

#include "Matrix.h"

using namespace std;

Matrix::Matrix(int row, int col) : rows(row), cols(col), data(row * col, 0.0) {}

double &Matrix::operator()(int r, int c)
{
    return data[cols * r + c];
}

const double &Matrix::operator()(int r, int c) const
{
    return data[cols * r + c];
}

Matrix Matrix::operator*(const Matrix &other) const
{
    if (cols != other.getRows())
    {
        throw invalid_argument("Dimension mismatch");
    }

    int c2 = other.getCols();
    Matrix ans(rows, c2);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < c2; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < cols; k++)
            {
                sum += (*this)(i, k) * other(k, j);
            }
            ans(i, j) = sum;
        }
    }
    return ans;
}

int Matrix::getRows() const
{
    return rows;
}

int Matrix::getCols() const
{
    return cols;
}

void Matrix::print() const
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            cout << data[cols * i + j] << ' ';
        }
        cout << endl;
    }
}

Matrix Matrix::transpose() const
{
    Matrix ans(cols, rows);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            ans(j, i) = (*this)(i, j);
        }
    }
    return ans;
}

Matrix Matrix::operator+(const Matrix &other) const
{
    if (rows != other.getRows() || cols != other.getCols())
    {
        throw invalid_argument("Dimension mismatch");
    }

    Matrix ans(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            ans(i, j) = (*this)(i, j) + other(i, j);
        }
    }
    return ans;
}

Matrix Matrix::operator-(const Matrix &other) const
{
    if (rows != other.getRows() || cols != other.getCols())
    {
        throw invalid_argument("Dimension mismatch");
    }

    Matrix ans(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            ans(i, j) = (*this)(i, j) - other(i, j);
        }
    }
    return ans;
}

Matrix Matrix::operator*(double scalar) const
{
    Matrix ans(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            ans(i, j) = scalar * (*this)(i, j);
        }
    }
    return ans;
}

Matrix &Matrix::operator+=(const Matrix &other)
{
    if (rows != other.getRows() || cols != other.getCols())
    {
        throw invalid_argument("Dimension mismatch");
    }

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            (*this)(i, j) += other(i, j);
        }
    }
    return *this;
}

Matrix &Matrix::operator-=(const Matrix &other)
{
    if (rows != other.getRows() || cols != other.getCols())
    {
        throw invalid_argument("Dimension mismatch");
    }

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            (*this)(i, j) -= other(i, j);
        }
    }
    return *this;
}

Matrix Matrix::hadamard(const Matrix &other) const
{
    if (rows != other.getRows() || cols != other.getCols())
    {
        throw invalid_argument("Dimension mismatch");
    }

    Matrix ans(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            ans(i, j) = (*this)(i, j) * other(i, j);
        }
    }
    return ans;
}

void Matrix::randomize(double mean, double stddev)
{
    random_device rd;  // random_device takes a lot of time so it is used only to generate a random seed
    mt19937 gen(rd()); // generates random bits producing numbers b/w 0 and 2^32
    normal_distribution<double> dist(mean, stddev);

    for (int i = 0; i < data.size(); i++)
    {
        data[i] = dist(gen); // dist takes the numbers from gen and convert them to those which fall on the gaussian curve
    }
}

void Matrix::fill(double value)
{
    int n = data.size();
    for (int i = 0; i < n; i++)
    {
        data[i] = value;
    }
}
