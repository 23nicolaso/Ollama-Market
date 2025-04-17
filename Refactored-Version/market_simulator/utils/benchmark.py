from reworkedOB import OrderBook as reob
# from order_book import OrderBook as ob
from market_simulator.utils.market_utils import accounts
from account import Account
import time
import statistics
import random
import cProfile

# Run multiple iterations to get average performance

def testReob():
    iterations = 50000
    times = []

    book = reob("SPY", 500)
    accounts.clear()
    account1 = Account("john", 100000)
    account2 = Account("bob", 100000)
    accounts["john"] = account1
    accounts["bob"] = account2

    for _ in range(iterations):
        start = time.perf_counter()
        
        book.addOrder("buy", round(random.random()*400,2), random.randint(1,1000), "limit", "john")
        book.addOrder("sell", round(random.random()*400,2), random.randint(1,1000), "limit", "bob")
        book.addOrder("sell", 50, random.randint(1,1000), "market", "bob")
        book.addOrder("buy", 50, random.randint(1,1000), "market", "john")
        
        end = time.perf_counter()
        times.append(end - start)

    # Calculate statistics
    avg_time = statistics.mean(times)/4
    ops_per_sec = iterations*4 / sum(times)

    print(f"\nPerformance metrics:")
    print(f"Average time per iteration: {avg_time*1000:.3f} ms")
    print(f"Operations per second: {ops_per_sec:.1f}")

def testOb():
    iterations = 50000
    times = []

    book = ob("SPY", 500)
    accounts.clear()
    account1 = Account("john", 100000)
    account2 = Account("bob", 100000)
    accounts["john"] = account1
    accounts["bob"] = account2

    for _ in range(iterations):
        start = time.perf_counter()
        
        book.addOrder("buy", round(random.random()*400,2), random.randint(1,1000), "limit", "john")
        book.addOrder("sell", round(random.random()*400,2), random.randint(1,1000), "limit", "bob")
        book.addOrder("sell", 50, random.randint(1,1000), "market", "bob")
        book.addOrder("buy", 50, random.randint(1,1000), "market", "john")
        
        end = time.perf_counter()
        times.append(end - start)

    # Calculate statistics
    avg_time = statistics.mean(times)/4
    ops_per_sec = iterations*4 / sum(times)

    print(f"\nPerformance metrics:")
    print(f"Average time per iteration: {avg_time*1000:.3f} ms")
    print(f"Operations per second: {ops_per_sec:.1f}")

cProfile.run('testReob()', 'reobProfile.prof')