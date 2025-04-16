from pstats import Stats

# Load the profile
stats = Stats('cyobProfile.prof')

# Sort and print to text
stats.strip_dirs().sort_stats('cumulative').print_stats(50)

# Load the profile
stats = Stats('reobProfile.prof')

# Sort and print to text
stats.strip_dirs().sort_stats('cumulative').print_stats(50)

# Load the profile
stats = Stats('obProfile.prof')

# Sort and print to text
stats.strip_dirs().sort_stats('cumulative').print_stats(50)
