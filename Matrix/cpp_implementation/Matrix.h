#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <functional>

class Matrix
{
public:
    Matrix(int rows, int cols, bool init_zero = true);
    Matrix(const std::vector<std::vector<double>> &data);
    ~Matrix();

    // Copy constructor
    Matrix(const Matrix &other);
    // Move constructor
    Matrix(Matrix &&other) noexcept;
    // Assignment operator
    Matrix &operator=(const Matrix &other);
    // Move assignment
    Matrix &operator=(Matrix &&other) noexcept;

    int getRows() const;
    int getCols() const;
    double &at(int row, int col);
    const double &at(int row, int col) const;

    Matrix multiply(const Matrix &other) const;
    Matrix add(const Matrix &other) const;
    Matrix transpose() const;

    void print() const;

    // Expose raw data for advanced usage if needed
    const double *getData() const;
    double *getData();

    // Force evaluation of lazy operations
    void evaluate() const;

private:
    int rows;
    int cols;
    double *data; // Aligned raw pointer

    // Lazy evaluation support
    mutable bool is_lazy = false;
    mutable std::function<void(Matrix &)> lazy_computation;
};

#endif // MATRIX_H
