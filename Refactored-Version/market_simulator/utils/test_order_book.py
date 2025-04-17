"""
A series of tests to ensure that @reworked_order_book is working effectively
"""

from market_simulator.models.diffOrderbook import OrderBook
from market_simulator.models.account import Account
from market_simulator.utils.market_utils import accounts

def remake():
    # reset accounts for next test set
    ob = OrderBook("SPY", 500)
    accounts.clear()
    account1 = Account(0, 100000)
    account2 = Account(1, 100000)
    accounts[0] = account1
    accounts[1] = account2
    return ob, account1, account2

def displayTraders(account1, account2):
    return
    # print("TRADER A: ", account1.getPosition("SPY"), account1.getPosition("CASH"), " TRADER B: ", account2.getPosition("SPY"), account2.getPosition("CASH"))

# buy 500, buy 510, buy 515, sell 0.5 lots at 515, sell 510, sell 530
# Expected 515*100+510*50=$77000 cash change, 100-500,50-510, sell 100-530
def main():
    # ob, account1, account2 = remake()
    # ob.addOrder(1, 500, 100, True, 0)
    # ob.addOrder(1, 510, 100, True, 0)
    # ob.addOrder(1, 515, 100, True, 0)
    # #ob.display()
    # displayTraders(account1, account2)

    # ob.addOrder(-1, 515, 50, True, 1)
    # #ob.display()
    # displayTraders(account1, account2)
    # ob.addOrder(-1, 510, 100, True, 1)
    # #ob.display()
    # displayTraders(account1, account2)
    # ob.addOrder(-1, 530, 100, True, 1)
    # ob.addOrder(-1, 540, 100, True, 1)
    # #ob.display()
    # displayTraders(account1, account2)
    # ob.addOrder(1, 510, 1, False, 1)
    # #ob.display()
    # displayTraders(account1, account2)
    # ob.addOrder(-1, 530, 1, False, 1)
    # #ob.display()
    # displayTraders(account1, account2)

    # bid 500 ask 500 clear @ 500
    print("\n BUY 500, SELL 500. FULL FILL @ 500")
    ob, account1, account2 = remake()
    ob.addOrder(1, 500, 100, True, 0)
    #ob.display()
    displayTraders(account1, account2)
    ob.addOrder(-1, 500, 100, True, 1)
    #ob.display()
    displayTraders(account1, account2)

    print("\n SELL 500, BUY 500. FULL FILL @ 500")
    ob, account1, account2 = remake()
    ob.addOrder(-1, 500, 100, True, 0)
    #ob.display()
    displayTraders(account1, account2)
    ob.addOrder(1, 500, 100, True, 1)
    #ob.display()
    displayTraders(account1, account2)


    # buy 450 sell 550, no fill
    print("\n BUY 450, SELL 550. NO FILL")
    ob, account1, account2 = remake()
    ob.addOrder(1, 450, 100, True, 0)
    #ob.display()
    displayTraders(account1, account2)
    ob.addOrder(-1, 550, 100, True, 1)
    #ob.display()
    displayTraders(account1, account2)

    # buy 500 sell 450, full fill 
    print("\n BUY 500, SELL 450. FULL FILL")
    ob, account1, account2 = remake()
    ob.addOrder(1, 500, 100, True, 0)
    #ob.display()
    displayTraders(account1, account2)
    order2 = ob.addOrder(-1, 450, 100, True, 1)
    #ob.display()
    displayTraders(account1, account2)

    # sell 550 buy 600, full fill
    print("\n SELL 550, BUY 600. FULL FILL")
    ob, account1, account2 = remake()
    ob.addOrder(-1, 550, 100, True, 0)
    #ob.display()
    displayTraders(account1, account2)
    ob.addOrder(1, 600, 100, True, 1)
    #ob.display()
    displayTraders(account1, account2)

    # buy 450 and buy 400, sell 500 and sell 450, fill 450 level, others remain
    # print("\n BUY 450 & BUY 400, SELL 500 & SELL 450, fully filled 450, remaining at 400, 500")
    # ob, account1, account2 = remake()
    # import time
    # import statistics

    # # Run multiple iterations to get average performance
    # iterations = 10000
    # times = []

    # for _ in range(iterations):
    #     start = time.perf_counter()
        
    #     ob.addOrder(1, 450, 100, True, 0)
    #     ob.addOrder(1, 400, 100, True, 0)
    #     ob.addOrder(-1, 500, 100, True, 1) 
    #     ob.addOrder(-1, 450, 100, True, 1)
        
    #     end = time.perf_counter()
    #     times.append(end - start)

    # # Calculate statistics
    # avg_time = statistics.mean(times)/4
    # ops_per_sec = iterations*4 / sum(times)

    # print(f"\nPerformance metrics:")
    # print(f"Average time per iteration: {avg_time*1000:.3f} ms")
    # print(f"Operations per second: {ops_per_sec:.1f}")

    # # Display final state
    # #ob.display()
    # displayTraders(account1, account2)

    # limit buy and market sell
    print("\n LIM BUY & MARKET SELL, EXPECTED 1 full bid order @ 500, 100 sold @ 550")
    ob, account1, account2 = remake()
    ob.addOrder(1, 500, 100, True, 0)
    ob.addOrder(1, 550, 100, True, 0)
    #ob.display()
    displayTraders(account1, account2)
    order2 = ob.addOrder(-1, 0, 50, False, 1)
    #ob.display()
    displayTraders(account1, account2)
    order2 = ob.addOrder(-1, 600, 50, False, 1)
    #ob.display()
    displayTraders(account1, account2)

    # limit sell and market buy
    print("\n LIM SELL & MARKET BUY, EXPECTED 1 full ask order @ 600, 100 sold @ 550")
    ob, account1, account2 = remake()
    ob.addOrder(-1, 550, 100, True, 0)
    ob.addOrder(-1, 600, 100, True, 0)
    #ob.display()
    displayTraders(account1, account2)
    ob.addOrder(1, 600, 50, False, 1)
    ob.addOrder(1, 0, 50, False, 1)
    #ob.display()
    displayTraders(account1, account2)

    # market buy and market sell
    print("\n MARKET SELL & MARKET BUY")
    ob, account1, account2 = remake()
    ob.addOrder(-1, 0, 100, False, 0)
    #ob.display()
    displayTraders(account1, account2)
    ob.addOrder(1, 0, 100, False, 1)
    #ob.display()
    displayTraders(account1, account2)

    # market buy and market sell with existing limit orders
    print("\n MARKET BUY & MARKET SELL, EXPECTED 0 ORDERS REMAINING")
    ob, account1, account2 = remake()
    ob.addOrder(-1, 600, 100, True, 0)
    ob.addOrder(1, 400, 100, True, 1)
    ob.addOrder(-1, 0, 100, False, 0)
    #ob.display()
    displayTraders(account1, account2)
    ob.addOrder(1, 0, 100, False, 1)
    #ob.display()
    displayTraders(account1, account2)

    # market buy and then limit sell
    print("\n MARKET BUY & LIMIT SELL")
    ob, account1, account2 = remake()
    ob.addOrder(1, 0, 100, False, 0)
    #ob.display()
    displayTraders(account1, account2)
    ob.addOrder(-1, 500, 100, True, 1)
    #ob.display()
    displayTraders(account1, account2)

    # market sell and then limit buy
    print("\n MARKET SELL & LIMIT BUY")
    ob, account1, account2 = remake()
    ob.addOrder(-1, 0, 100, False, 0)
    #ob.display()
    displayTraders(account1, account2)
    ob.addOrder(1, 500, 100, True, 1)
    #ob.display()
    displayTraders(account1, account2)

    ob, account1, account2 = remake()
    ord = ob.addOrder(1, 100, 100, True, 0)
    # status = ob.cancelOrder(ord)
    # print(status)

    # from market_simulator.agents.market_maker import MarketMaker
    # from market_simulator.config import SPREADS
    # ob,acc1,acc2 = remake()
    # mm = MarketMaker("testacc", 100000000, SPREADS)
    # mm.makeMarket(ob)
    # #ob.display()
    # ob.addOrder(1, 600, 100000, False, 0)
    # #ob.display()
    # mm.provideLiquidity(ob)
    # #ob.display()
    # mm.makeMarket(ob)
    # #ob.display()
    # print(ob.getNearbyDepth(0.1))
main()