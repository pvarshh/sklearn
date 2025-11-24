#ifndef KNN_H
#define KNN_H

#include <vector>
#include <string>
#include <memory>
#include <queue>

struct KDNode {
    std::vector<int> point;
    std::string label;
    std::shared_ptr<KDNode> left;
    std::shared_ptr<KDNode> right;
    int axis;

    KDNode(std::vector<int> pt, std::string lbl, int ax)
        : point(std::move(pt)), label(std::move(lbl)), left(nullptr), right(nullptr), axis(ax) {}
};

class KNN {
private:
    int k;
    std::shared_ptr<KDNode> root;
    size_t num_features;

    // Helper to build the tree
    std::shared_ptr<KDNode> buildTree(std::vector<std::pair<std::vector<int>, std::string>> &data, int depth);

    // Helper to search the tree
    void searchRecursive(const std::shared_ptr<KDNode> &node, const std::vector<int> &target,
                         std::priority_queue<std::pair<long long, std::string>> &pq) const;

public:
    KNN(int k = 3);
    void fit(const std::string &filepath);
    std::string classify(const std::vector<int> &point) const;
    std::vector<std::string> classify(const std::vector<std::vector<int>> &points) const;
};

#endif
