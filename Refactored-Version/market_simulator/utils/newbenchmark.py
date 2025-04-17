import time
import statistics
import cProfile
from random import getrandbits, randint

from market_simulator.models.sortedListOB import OrderBook, LimitOrder, MarketOrder
from market_simulator.utils.market_utils import accounts
from market_simulator.models.account import Account

def testOB():
    book = OrderBook("SPY", 500)
    accounts.clear()
    account1 = Account(0, 100000)
    account2 = Account(1, 100000)
    accounts[0] = account1
    accounts[1] = account2

    numOrders = 10**6
    orders = []
    for n in range(numOrders):
        if bool(getrandbits(1)):
            orders.append(LimitOrder(n, 0, randint(1, 200), randint(1, 4), 1))
        else:
            orders.append(LimitOrder(n, 1, randint(1, 200), randint(1, 4), -1))

    from time import time
    start = time()
    for order in orders:
        book.process_order(order)
    end = time()
    totalTime = (end-start)
    print("Time: " + str(totalTime))
    print("Time per order (us): " + str(1000000*totalTime/numOrders))
    print("Orders per second: " + str(numOrders/totalTime))
cProfile.run('testOB()', 'newOB.prof')