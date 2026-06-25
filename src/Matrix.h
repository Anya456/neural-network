#include <iostream>
#include <vector>

using namespace std;

class Matrix
{
private:
    int rows;
    int cols;
    vector<double> data;

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
    Matrix relu() const;
    Matrix reluDerivative() const;

    void fill(double value);
    void randomize(double mean, double stddev);
    void print();

    int getRows() const;
    int getCols() const;
};