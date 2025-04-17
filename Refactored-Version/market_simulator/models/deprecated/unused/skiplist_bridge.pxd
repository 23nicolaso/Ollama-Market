# distutils: language = c++

from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "skiplist.h":
    cdef cppclass SkipList:
        SkipList(bool sortAscending) except +
        bool contains(double key) 
        void insert(double key)
        void clear()
        void erase_up_to(double max_key, bint inclusive)
        void get(vector[double]& result)
        void remove(double key)
        double getFirst() 
        void print()
        void get_up_to(double key, vector[double]& result, bint inclusive)