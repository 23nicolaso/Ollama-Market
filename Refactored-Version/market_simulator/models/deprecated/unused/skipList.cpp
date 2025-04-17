#include "skiplist.h"

int SkipList::randomLevel() {
    int lvl = 0;
    while (((double)std::rand() / RAND_MAX) < P && lvl < MAX_LEVEL)
        lvl++;
    return lvl;
}

SkipList::SkipList(bool sortAscending) : sortAscending(sortAscending) {
    level = 0;
    header = new Node(sortAscending ? -INFINITY : INFINITY, MAX_LEVEL);
    std::srand(static_cast<unsigned int>(std::time(nullptr))); // seed for randomness
}

SkipList::~SkipList() {
    Node* current = header;
    while (current) {
        Node* next = current->forward[0];
        delete current;
        current = next;
    }
}

bool SkipList::contains(double key) const {
    Node* current = header;
    for (int i = level; i >= 0; i--) {
        while (current->forward[i] && 
               (sortAscending ? current->forward[i]->key < key : current->forward[i]->key > key)) {
            current = current->forward[i];
        }
    }
    current = current->forward[0];
    return current && current->key == key;
}

double SkipList::getFirst() const {
    if (header->forward[0]) {
        return header->forward[0]->key;
    }
    else{
        return -1;
    }
}

void SkipList::clear() {
    Node* current = header->forward[0];
    while (current) {
        Node* next = current->forward[0];
        delete current;
        current = next;
    }
    
    // Reset the header's forward pointers
    for (int i = 0; i <= MAX_LEVEL; i++) {
        header->forward[i] = nullptr;
    }
    
    // Reset the level
    level = 0;
}

void SkipList::insert(double key) {
    std::vector<Node*> update(MAX_LEVEL + 1);
    Node* current = header;

    for (int i = level; i >= 0; i--) {
        while (current->forward[i] && 
               (sortAscending ? current->forward[i]->key < key : current->forward[i]->key > key)) {
            current = current->forward[i];
        }
        update[i] = current;
    }

    current = current->forward[0];
    if (current && current->key == key) return; // Already present

    int newLevel = randomLevel();
    if (newLevel > level) {
        for (int i = level + 1; i <= newLevel; i++) {
            update[i] = header;
        }
        level = newLevel;
    }

    Node* newNode = new Node(key, newLevel);
    for (int i = 0; i <= newLevel; i++) {
        newNode->forward[i] = update[i]->forward[i];
        update[i]->forward[i] = newNode;
    }
}

void SkipList::erase_up_to(double max_key, bool inclusive) {
    Node* current = header;
    
    // For descending order, we erase elements greater than or equal to max_key
    // For ascending order, we erase elements less than or equal to max_key
    while (current->forward[0] && 
           (sortAscending ? (inclusive? current->forward[0]->key <= max_key : current->forward[0]->key < max_key) :
                            (inclusive? current->forward[0]->key >= max_key : current->forward[0]->key > max_key)
            )
     ) {
        Node* to_delete = current->forward[0];
        for (int i = 0; i <= level; i++) {
            if (current->forward[i] != to_delete) break;
            current->forward[i] = to_delete->forward[i];
        }
        delete to_delete;
    }

    // Adjust the level
    while (level > 0 && header->forward[level] == nullptr)
        level--;
}

void SkipList::get(std::vector<double>& result) {
    result.clear();
    Node* current = header->forward[0];
    while (current) {
        result.push_back(current->key);
        current = current->forward[0];
    }
}

void SkipList::get_up_to(double key, std::vector<double>& result, bool inclusive) const {
    result.clear();
    
    Node* current = header->forward[0];
    
    while (current) {
        if (sortAscending) {
            if (inclusive ? current->key > key : current->key >= key) {
                break;
            }
        } else {  // Descending
            if (inclusive ? current->key < key : current->key <= key) {
                break;
            }
        }
        
        result.push_back(current->key);
        current = current->forward[0];
    }
}

void SkipList::print() {
    Node* current = header->forward[0];
    std::cout << "SkipList (" << (sortAscending ? "Ascending" : "Descending") << "): ";
    while (current) {
        std::cout << current->key << " ";
        current = current->forward[0];
    }
    std::cout << "\n";
}

void SkipList::remove(double key) {
    std::vector<Node*> update(MAX_LEVEL + 1);
    Node* current = header;

    // Traverse down from the top level to find nodes that need updating
    for (int i = level; i >= 0; i--) {
        while (current->forward[i] &&
               (sortAscending ? current->forward[i]->key < key : current->forward[i]->key > key)) {
            current = current->forward[i];
        }
        update[i] = current;
    }

    // Move to the target node
    current = current->forward[0];

    // Check if the key matches
    if (current && current->key == key) {
        // Adjust pointers at each level
        for (int i = 0; i <= level; i++) {
            if (update[i]->forward[i] != current)
                break;
            update[i]->forward[i] = current->forward[i];
        }

        delete current;

        // Adjust the current max level of the skip list
        while (level > 0 && header->forward[level] == nullptr)
            level--;
    }
}
