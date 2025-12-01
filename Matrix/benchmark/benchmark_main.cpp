#include "../cpp_implementation/Matrix.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <string>

std::vector<std::vector<double>> read_csv(const std::string &filepath)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    std::vector<std::vector<double>> data;
    std::string line;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string val_str;
        std::vector<double> row;
        while (std::getline(ss, val_str, ','))
        {
            row.push_back(std::stod(val_str));
        }
        data.push_back(row);
    }
    return data;
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <matrix_A_path> <matrix_B_path>" << std::endl;
        return 1;
    }

    std::string matrix_a_path = argv[1];
    std::string matrix_b_path = argv[2];

    std::cout << "Reading matrices..." << std::endl;
    auto data_A = read_csv(matrix_a_path);
    auto data_B = read_csv(matrix_b_path);

    Matrix A(data_A);
    Matrix B(data_B);

    std::cout << "Matrix size: " << A.getRows() << "x" << A.getCols() << std::endl;

    // Benchmark Multiplication
    auto start = std::chrono::high_resolution_clock::now();
    Matrix C = A.multiply(B);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Multiplication Time: " << diff.count() << " s" << std::endl;

    // Benchmark Addition
    start = std::chrono::high_resolution_clock::now();
    Matrix D = A.add(B);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Addition Time: " << diff.count() << " s" << std::endl;

    // Benchmark Transpose
    start = std::chrono::high_resolution_clock::now();
    Matrix E = A.transpose();
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Transpose Time: " << diff.count() << " s" << std::endl;

    return 0;
}
