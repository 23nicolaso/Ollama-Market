import cProfile
import test_order_book as tb

def my_function():
    # Your code here
    tb.main()

cProfile.run('my_function()', 'profile_output_reworkedorderbook.prof')
