# from pyskiplist import PySkipList

# # Create a new skip list
# sl = PySkipList(False)

# # Insert some values
# sl.insert(5.0)
# sl.insert(10.0)
# sl.insert(7.5)
# sl.insert(10.0)

# print(sl.getFirst())
# # Check if a value exists
# print(sl.contains(7.5))  # True
# print(sl.contains(8.0))  # False

# # Print the list
# sl.print_list()

# # Remove values up to a threshold
# sl.erase_up_to(7.5, False)
# sl.print_list()

# sl.insert(1.2)
# sl.insert(1.3)

# sl.print_list()
# print(sl.get_up_to(0))

# sl.clear()
# sl.insert(1.2)
# sl.insert(1.3)
# sl.print_list()
# sl.remove(1.3)
# sl.print_list()

from skipList import SkipListSet

sls_asc = SkipListSet[int](ascending=True)
keys_to_insert = [3, 6, 7, 9, 12, 19, 17, 26, 21, 25, 1, 5, 2]
insert_results = [sls_asc.insert(k) for k in keys_to_insert]
print(f"Insertion results (all should be True): {all(insert_results)}")
print(f"Try inserting duplicate 6: {sls_asc.insert(6)}") # Should be False

print(sls_asc)
print(f"Length: {len(sls_asc)}")
print(f"First key: {sls_asc.get_first()}")
print(f"Contains 17? {sls_asc.contains(17)}")
print(f"Contains 10? {sls_asc.contains(10)}")
print(f"9 in set? {9 in sls_asc}")
print(f"10 in set? {10 in sls_asc}")

print("\nKeys before 10 (exclusive):")
print(sls_asc.get_before(10, inclusive=False))

print("\nKeys before 12 (inclusive):")
print(sls_asc.get_before(12, inclusive=True))

print("\nRemoving keys before 7 (exclusive)...")
removed_count = sls_asc.remove_before(7, inclusive=False)
print(f"Removed {removed_count} keys.")
print(sls_asc)
print(f"First key now: {sls_asc.get_first()}")
print(f"Length now: {len(sls_asc)}")
print(f"Contains 1? {1 in sls_asc}") # Should be False

print("\nRemoving keys before 20 (inclusive)...")
removed_count = sls_asc.remove_before(20, inclusive=True)
print(f"Removed {removed_count} keys.")
print(sls_asc)
print(f"First key now: {sls_asc.get_first()}")
print(f"Length now: {len(sls_asc)}")

print("\nIterating over keys:")
for k in sls_asc:
    print(f"  {k}")

print("\nDeleting 25:")
delete_result = sls_asc.delete(25)
print(f"Delete successful? {delete_result}")
print(f"Delete non-existent 99: {sls_asc.delete(99)}") # Should be False
print(sls_asc)

print("\n--- Descending Skip List Set ---")
sls_desc = SkipListSet[str](ascending=False)
sls_desc.insert("apple")
sls_desc.insert("zebra")
sls_desc.insert("banana")
sls_desc.insert("grape")
sls_desc.insert("orange")

print(sls_desc)
print(f"Length: {len(sls_desc)}")
print(f"First key: {sls_desc.get_first()}") # Should be zebra

print("\nKeys before 'grape' (exclusive, descending means > 'grape'):")
print(sls_desc.get_before("grape", inclusive=False)) # Should be zebra, orange

print("\nKeys before 'banana' (inclusive, descending means >= 'banana'):")
print(sls_desc.get_before("banana", inclusive=True)) # Should be zebra, orange, grape, banana

print("\nRemoving keys before 'orange' (exclusive, descending means > 'orange')...")
removed_count = sls_desc.remove_before("orange", inclusive=False) # Remove zebra
print(f"Removed {removed_count} keys.")
print(sls_desc)
print(f"First key now: {sls_desc.get_first()}") # Should be orange