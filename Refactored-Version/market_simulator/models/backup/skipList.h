#ifndef SKIPLIST_H
#define SKIPLIST_H

#include <vector>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <ctime>
#include <iostream>

class SkipList {
    struct Node {
        double key;
        std::vector<Node*> forward;

        Node(double k, int level) : key(k), forward(level + 1, nullptr) {}
    };

    int level;
    Node* header;
    const int MAX_LEVEL = 10;
    const float P = 0.5f;
    bool sortAscending;

    int randomLevel();

public:
    SkipList(bool sortAscending = true);
    ~SkipList();
    
    bool contains(double key) const;
    void insert(double key);
    void clear();
    void erase_up_to(double max_key, bool inclusive = true);
    void get(std::vector<double>& result);
    void get_up_to(double key, std::vector<double>& result, bool inclusive = true) const;
    void remove(double key);
    double getFirst() const;
    void print();
};

#endif // SKIPLIST_H