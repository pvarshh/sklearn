#include "KNN.h"
#include <cmath>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <future>

KNN::KNN(int k) : k(k), root(nullptr), num_features(0) {}

// Helper function to calculate squared Euclidean distance
static long long squared_distance(
    const std::vector<int> &a,
    const std::vector<int> &b
) {
    long long dist = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        long long diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

std::shared_ptr<KDNode> KNN::buildTree(
    std::vector<std::pair<std::vector<int>,
    std::string>> &data,
    int depth
) {
    if (data.empty()) { return nullptr; }

    int axis = depth % num_features;
    size_t mid = data.size() / 2;

    // Sort data by the current axis
    std::nth_element(data.begin(), data.begin() + mid, data.end(),
                     [axis](const std::pair<std::vector<int>, std::string> &a, const std::pair<std::vector<int>, std::string> &b)
                     {
                         return a.first[axis] < b.first[axis];
                     });

    // Create node
    auto node = std::make_shared<KDNode>(data[mid].first, data[mid].second, axis);

    // Recursively build subtrees
    std::vector<std::pair<std::vector<int>, std::string>> left_data(data.begin(), data.begin() + mid);
    std::vector<std::pair<std::vector<int>, std::string>> right_data(data.begin() + mid + 1, data.end());

    node->left = buildTree(left_data, depth + 1);
    node->right = buildTree(right_data, depth + 1);

    return node;
}



void KNN::searchRecursive(
    const std::shared_ptr<KDNode> &node, 
    const std::vector<int> &target,
    std::priority_queue<std::pair<long long,
    std::string>> &pq
) const {
    if (!node) { return; }

    long long dist = squared_distance(target, node->point);

    // Maintain k nearest neighbors in the priority queue
    if (pq.size() < static_cast<size_t>(k)) {
        pq.push({dist, node->label});
    }
    else if (dist < pq.top().first) {
        pq.pop();
        pq.push({dist, node->label});
    }

    // Determine which subtree to search first
    int axis = node->axis;
    long long diff = target[axis] - node->point[axis];

    std::shared_ptr<KDNode> near = (diff < 0) ? node->left : node->right;
    std::shared_ptr<KDNode> far = (diff < 0) ? node->right : node->left;

    searchRecursive(near, target, pq);

    // Check if we need to search the far subtree
    // If we don't have k neighbors yet, or if the distance to the splitting plane is less than the worst distance in our current k-best
    if (pq.size() < static_cast<size_t>(k) || (diff * diff) < pq.top().first) {
        searchRecursive(far, target, pq);
    }
}

std::string KNN::classify(
    const std::vector<int> &point
) const {
    if (point.size() != num_features) { throw std::invalid_argument("Query point has different number of features than training data."); }

    // Max-heap to store k nearest neighbors (distance, label)
    std::priority_queue<std::pair<long long, std::string>> pq;

    searchRecursive(root, point, pq);

    // Count votes
    std::map<std::string, int> votes;
    while (!pq.empty()) {
        votes[pq.top().second]++;
        pq.pop();
    }

    // Find max votes
    std::string best_label;
    int max_votes = -1;
    for (const auto &pair : votes) {
        if (pair.second > max_votes) {
            max_votes = pair.second;
            best_label = pair.first;
        }
    }

    return best_label;
}

std::vector<std::string> KNN::classify(
    const std::vector<std::vector<int>> &points
) const {
    std::vector<std::string> results(points.size());

    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 4; // Default fallback
    }

    std::vector<std::future<void>> futures;
    size_t batch_size = (points.size() + num_threads - 1) / num_threads;

    for (unsigned int i = 0; i < num_threads; ++i) {
        size_t start = i * batch_size;
        size_t end = std::min(start + batch_size, points.size());

        if (start >= end) { break; }

        futures.push_back(std::async(std::launch::async, [this, &points, &results, start, end]() {
            for (size_t j = start; j < end; ++j) {
                results[j] = this->classify(points[j]);
            } }));
    }

    for (auto &f : futures) { f.get(); }

    return results;
}

void KNN::fit(
    const std::string &filepath
) {
    std::ifstream file(filepath);
    if (!file.is_open()) { throw std::runtime_error("Could not open file: " + filepath); }

    std::string line;
    // 1. Skip metadata (1st row)
    if (!std::getline(file, line)) { throw std::runtime_error("File is empty or missing metadata row."); }

    std::vector<std::pair<std::vector<int>, std::string>> data;

    // 2. Read data rows
    while (std::getline(file, line)) {
        if (line.empty()) { continue; }

        std::stringstream ss(line);
        std::string val_str;
        std::vector<int> features;
        std::string label;
        std::vector<std::string> row_values;

        // Parse comma-separated values
        while (std::getline(ss, val_str, ',')) {
            // Trim whitespace
            val_str.erase(0, val_str.find_first_not_of(" \t\r\n"));
            val_str.erase(val_str.find_last_not_of(" \t\r\n") + 1);
            if (!val_str.empty()) {
                row_values.push_back(val_str);
            }
        }

        // Fallback: Try space-separated if CSV parsing yielded too few columns
        if (row_values.size() <= 1 && line.find(',') == std::string::npos) {
            row_values.clear();
            std::stringstream ss2(line);
            while (ss2 >> val_str) {
                row_values.push_back(val_str);
            }
        }

        if (row_values.size() < 2) { continue; }

        // Last element is label
        label = row_values.back();
        row_values.pop_back();

        // Rest are features
        for (const auto &v : row_values) {
            try {
                features.push_back(std::stoi(v));
            }
            catch (...) {
                throw std::runtime_error("Invalid feature value: " + v);
            }
        }

        if (num_features == 0) {
            num_features = features.size();
        }
        else if (features.size() != num_features) { throw std::runtime_error("Inconsistent feature size in data file."); }

        data.emplace_back(features, label);
    }

    if (data.empty()) { throw std::runtime_error("No valid data found in file."); }

    root = buildTree(data, 0);
}
