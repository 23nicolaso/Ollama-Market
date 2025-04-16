# distutils: language = c++

from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "skiplist.h":
    cdef cppclass SkipList:
        SkipList(bool sortAscending) except +
        bool contains(double key) 
        void insert(double key)
        void erase_up_to(double max_key, bint inclusive)
        void print()
        void clear()
        void get(vector[double]& result)
        double getFirst() const
        void get_up_to(double key, vector[double]& result, bint inclusive)
        void remove(double key)

cdef class PySkipList:
    """
    Python wrapper for the C++ SkipList class.
    
    This class provides an efficient implementation of the Skip List data structure,
    a probabilistic alternative to balanced trees.
    
    Parameters:
        sort_ascending (bool): If True, the list is sorted in ascending order.
                               If False, the list is sorted in descending order.
    """
    cdef SkipList* _thisptr
    
    def __cinit__(self, bool sort_ascending=True):
        self._thisptr = new SkipList(sort_ascending)
        
    def __dealloc__(self):
        if self._thisptr != NULL:
            del self._thisptr
            
    def contains(self, double key):
        """
        Check if the skip list contains the given key.
        
        Args:
            key (float): The key to check for
            
        Returns:
            bool: True if the key exists in the skip list, False otherwise
        """
        return self._thisptr.contains(key)
        
    def insert(self, double key):
        """
        Insert a key into the skip list (if it doesn't already exist).
        
        Args:
            key (float): The key to insert
        """
        self._thisptr.insert(key)
        
    def erase_up_to(self, double max_key, bint inclusive=True):
        """
        Remove all keys up to the given max_key.
        
        In ascending mode: Remove all keys less than or equal to max_key.
        In descending mode: Remove all keys greater than or equal to max_key.
        
        Args:
            max_key (float): The threshold value for removal
        """
        self._thisptr.erase_up_to(max_key, inclusive)

    def clear(self):
        """
        Clear the contents of the skip list.
        """
        self._thisptr.clear()  
            
    def print_list(self):
        """
        Print the contents of the skip list to stdout.
        """
        self._thisptr.print()
    
    def get(self):
        """
        Get the contents of the whole skip list
        """
        cdef vector[double] result
        self._thisptr.get(result)
        # Convert C++ vector to Python list
        return [result[i] for i in range(result.size())]

    def get_up_to(self, double key, bint inclusive=True):
        """
        Get the contents of the skip list up to a given key.
        """
        cdef vector[double] result
        self._thisptr.get_up_to(key, result, inclusive)
        # Convert C++ vector to Python list
        return [result[i] for i in range(result.size())]

    def remove(self, double key):
        """
        Remove a given key from the skip list.
        """
        self._thisptr.remove(key)

    def getFirst(self) :
        """
        Get the first element in the skip list
        """
        return self._thisptr.getFirst()