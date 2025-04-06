"""
A series of tests to ensure that @reworked_order_book is working effectively
"""

from market_simulator.models.reworked_order_book import OrderBook
from market_simulator.models.account import Account
from market_simulator.utils.market_utils import accounts

def remake():
    # reset accounts for next test set
    ob = OrderBook("SPY", 500)
    accounts.clear()
    account1 = Account("john", 100000)
    account2 = Account("bob", 100000)
    accounts["john"] = account1
    accounts["bob"] = account2
    return ob, account1, account2

def displayTraders():
    print("TRADER A: ", account1.getPosition("SPY"), account1.getPosition("CASH"), " TRADER B: ", account2.getPosition("SPY"), account2.getPosition("CASH"))

# buy 500, buy 510, buy 515, sell 0.5 lots at 515, sell 510, sell 530
# Expected 515*100+510*50=$77000 cash change, 100-500,50-510, sell 100-530

# ob, account1, account2 = remake()
# ob.addOrder("buy", 500, 100, "limit", "john")
# ob.addOrder("buy", 510, 100, "limit", "john")
# ob.addOrder("buy", 515, 100, "limit", "john")
# ob.display()
# displayTraders()

# ob.addOrder("sell", 515, 50, "limit", "bob")
# ob.display()
# displayTraders()
# ob.addOrder("sell", 510, 100, "limit", "bob")
# ob.display()
# displayTraders()
# ob.addOrder("sell", 530, 100, "limit", "bob")
# ob.addOrder("sell", 540, 100, "limit", "bob")
# ob.display()
# displayTraders()

# ob.addOrder("buy", 510, 1, "market", "bob")
# ob.display()
# displayTraders()
# ob.addOrder("sell", 530, 1, "market", "bob")
# ob.display()
# displayTraders()

# bid 500 ask 500 clear @ 500
# print("\n BUY 500, SELL 500. FULL FILL @ 500")
# ob, account1, account2 = remake()
# ob.addOrder("buy", 500, 100, "limit", "john")
# ob.display()
# displayTraders()

# ob.addOrder("sell", 500, 100, "limit", "bob")
# ob.display()
# displayTraders()

# print("\n SELL 500, BUY 500. FULL FILL @ 500")
# ob, account1, account2 = remake()
# ob.addOrder("sell", 500, 100, "limit", "john")
# ob.display()
# displayTraders()

# ob.addOrder("buy", 500, 100, "limit", "bob")
# ob.display()
# displayTraders()


# buy 450 sell 550, no fill
# print("\n BUY 450, SELL 550. NO FILL")
# ob, account1, account2 = remake()
# ob.addOrder("buy", 450, 100, "limit", "john")
# ob.display()
# displayTraders()

# ob.addOrder("sell", 550, 100, "limit", "bob")
# ob.display()
# displayTraders()

# buy 500 sell 450, full fill 
# print("\n BUY 500, SELL 450. FULL FILL")
# ob, account1, account2 = remake()
# ob.addOrder("buy", 500, 100, "limit", "john")
# ob.display()
# displayTraders()
# order2 = ob.addOrder("sell", 450, 100, "limit", "bob")
# ob.display()
# displayTraders()

# # sell 550 buy 600, full fill
# print("\n SELL 550, BUY 600. FULL FILL")
# ob, account1, account2 = remake()
# ob.addOrder("sell", 550, 100, "limit", "john")
# ob.display()
# displayTraders()
# ob.addOrder("buy", 600, 100, "limit", "bob")
# ob.display()
# displayTraders()

# buy 450 and buy 400, sell 500 and sell 450, fill 450 level, others remain
print("\n BUY 450 & BUY 400, SELL 500 & SELL 450, fully filled 450, remaining at 400, 500")
ob, account1, account2 = remake()
import time
import statistics

# Run multiple iterations to get average performance
iterations = 1000
times = []

for _ in range(iterations):
    start = time.perf_counter()
    
    ob.addOrder("buy", 450, 100, "limit", "john")
    ob.addOrder("buy", 400, 100, "limit", "john")
    ob.addOrder("sell", 500, 100, "limit", "bob") 
    ob.addOrder("sell", 450, 100, "limit", "bob")
    
    end = time.perf_counter()
    times.append(end - start)

# Calculate statistics
avg_time = statistics.mean(times)
ops_per_sec = iterations / sum(times)

print(f"\nPerformance metrics:")
print(f"Average time per iteration: {avg_time*1000:.3f} ms")
print(f"Operations per second: {ops_per_sec:.1f}")

# Display final state
ob.display()
displayTraders()

# limit buy and market sell
# print("\n LIM BUY & MARKET SELL, EXPECTED 1 full bid order @ 500, 100 sold @ 550")
# ob, account1, account2 = remake()
# ob.addOrder("buy", 500, 100, "limit", "john")
# ob.addOrder("buy", 550, 100, "limit", "john")
# ob.display()
# displayTraders()
# order2 = ob.addOrder("sell", 0, 50, "market", "bob")
# ob.display()
# displayTraders()
# order2 = ob.addOrder("sell", 600, 50, "market", "bob")
# ob.display()
# displayTraders()

# # limit sell and market buy
# print("\n LIM SELL & MARKET BUY, EXPECTED 1 full ask order @ 600, 100 sold @ 550")
# ob, account1, account2 = remake()
# ob.addOrder("sell", 550, 100, "limit", "john")
# ob.addOrder("sell", 600, 100, "limit", "john")
# ob.display()
# displayTraders()
# ob.addOrder("buy", 600, 50, "market", "bob")
# ob.addOrder("buy", 0, 50, "market", "bob")
# ob.display()
# displayTraders()

# # market buy and market sell
# print("\n MARKET BUY & MARKET SELL")
# ob, account1, account2 = remake()
# ob.addOrder("sell", 0, 100, "market", "john")
# ob.display()
# displayTraders()
# ob.addOrder("buy", 0, 100, "market", "bob")
# ob.display()
# displayTraders()

# # market buy and market sell with existing limit orders
# print("\n MARKET BUY & MARKET SELL, EXPECTED 0 ORDERS REMAINING")
# ob, account1, account2 = remake()
# ob.addOrder("sell", 600, 100, "limit", "john")
# ob.addOrder("buy", 400, 100, "limit", "bob")
# ob.addOrder("sell", 0, 100, "market", "john")
# ob.display()
# displayTraders()
# ob.addOrder("buy", 0, 100, "market", "bob")
# ob.display()
# displayTraders()

# # market buy and then limit sell
# print("\n MARKET BUY & LIMIT SELL")
# ob, account1, account2 = remake()
# ob.addOrder("buy", 0, 100, "market", "john")
# ob.display()
# displayTraders()
# ob.addOrder("sell", 500, 100, "limit", "bob")
# ob.display()
# displayTraders()

# market sell and then limit buy
# print("\n MARKET SELL & LIMIT BUY")
# ob, account1, account2 = remake()
# ob.addOrder("sell", 0, 100, "market", "john")
# ob.display()
# displayTraders()
# ob.addOrder("buy", 500, 100, "limit", "bob")
# ob.display()
# displayTraders()

# ob, account1, account2 = remake()
# ord = ob.addOrder("buy", 100, 100, "limit", "john")
# status = ob.cancelOrder(ord)
# print(status)

# from market_simulator.agents.market_maker import MarketMaker
# from market_simulator.config import SPREADS
# ob,acc1,acc2 = remake()
# mm = MarketMaker("testacc", 100000000, SPREADS)
# mm.makeMarket(ob)
# ob.display()
# ob.addOrder("buy", 600, 100000, "market", "john")
# ob.display()
# mm.provideLiquidity(ob)
# ob.display()
# mm.makeMarket(ob)
# ob.display()
# print(ob.getNearbyDepth(0.1))