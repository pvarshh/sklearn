#include "../cpp_implementation/KNN.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <string>

// Helper to read test data (features only + optional label to ignore)
std::vector<std::vector<int>> read_test_data(const std::string &filepath)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open test file: " + filepath);
    }

    std::string line;
    // Skip header
    std::getline(file, line);

    std::vector<std::vector<int>> data;
    while (std::getline(file, line))
    {
        if (line.empty())
            continue;
        std::stringstream ss(line);
        std::string val_str;
        std::vector<int> features;
        std::vector<std::string> row_values;

        while (std::getline(ss, val_str, ','))
        {
            val_str.erase(0, val_str.find_first_not_of(" \t\r\n"));
            val_str.erase(val_str.find_last_not_of(" \t\r\n") + 1);
            if (!val_str.empty())
                row_values.push_back(val_str);
        }

        // If CSV failed, try space
        if (row_values.size() <= 1 && line.find(',') == std::string::npos)
        {
            row_values.clear();
            std::stringstream ss2(line);
            while (ss2 >> val_str)
            {
                row_values.push_back(val_str);
            }
        }

        if (row_values.size() < 2)
            continue;

        // Ignore last column (label)
        row_values.pop_back();

        for (const auto &v : row_values)
        {
            features.push_back(std::stoi(v));
        }
        data.push_back(features);
    }
    return data;
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <train_file> <test_file> <k>" << std::endl;
        return 1;
    }

    std::string train_file = argv[1];
    std::string test_file = argv[2];
    int k = std::stoi(argv[3]);

    try
    {
        KNN knn(k);

        // Measure Training Time
        auto start_train = std::chrono::high_resolution_clock::now();
        knn.fit(train_file);
        auto end_train = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> train_ms = end_train - start_train;

        // Load Test Data
        auto test_points = read_test_data(test_file);

        // Measure Single Prediction Time (Average of first 100)
        double single_pred_avg_ms = 0;
        int num_single_tests = std::min((int)test_points.size(), 100);
        if (num_single_tests > 0)
        {
            auto start_single = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < num_single_tests; ++i)
            {
                knn.classify(test_points[i]);
            }
            auto end_single = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end_single - start_single;
            single_pred_avg_ms = elapsed.count() / num_single_tests;
        }

        // Measure Batch Prediction Time
        auto start_pred = std::chrono::high_resolution_clock::now();
        auto predictions = knn.classify(test_points);
        auto end_pred = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> batch_pred_ms = end_pred - start_pred;

        // Output timings to stdout
        std::cout << train_ms.count() << std::endl;
        std::cout << single_pred_avg_ms << std::endl;
        std::cout << batch_pred_ms.count() << std::endl;

        // Output predictions to stdout
        for (const auto &p : predictions)
        {
            std::cout << p << "\n";
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
