from pstats import Stats

# Load the profile
stats = Stats('newOB.prof')

# Sort and print to text
stats.strip_dirs().sort_stats('cumulative').print_stats(50)
