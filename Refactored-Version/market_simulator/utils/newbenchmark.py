import time
import statistics
import cProfile
from random import getrandbits, randint

from market_simulator.models.sortedListOB import OrderBook, LimitOrder, MarketOrder, IcebergOrder
from market_simulator.utils.market_utils import accounts
from market_simulator.models.account import Account

# def testOB():
#     book = OrderBook("SPY", 500)
#     accounts.clear()
#     account1 = Account(0, 100000)
#     account2 = Account(1, 100000)
#     accounts[0] = account1
#     accounts[1] = account2

#     numOrders = 10**6
#     orders = []
#     for n in range(numOrders):
#         if bool(getrandbits(1)):
#             orders.append(LimitOrder(n, 0, randint(1, 200), randint(1, 4), 1))
#         else:
#             orders.append(LimitOrder(n, 1, randint(1, 200), randint(1, 4), -1))

#     from time import time
#     start = time()
#     for order in orders:
#         book.process_order(order)
#     end = time()
#     totalTime = (end-start)
#     print("Time: " + str(totalTime))
#     print("Time per order (us): " + str(1000000*totalTime/numOrders))
#     print("Orders per second: " + str(numOrders/totalTime))
# cProfile.run('testOB()', 'newOB.prof')

book = OrderBook("SPY", 500)
accounts.clear()
account_1 = Account(0, 100000)
account_2 = Account(1, 100000)
accounts[0] = account_1
accounts[1] = account_2
order_1 = IcebergOrder(0, 1100, 99.8, 1, 100)
order_2 = LimitOrder(1, 10, 99.8, -1)
order_3 = LimitOrder(1, 10, 99.8, -1)
# order_3 = LimitOrder(1, 10, 99.9, -1)
# order_3 = LimitOrder(0, 20, 100.1, 1)
# order_4 = LimitOrder(1, 20, 100.2, -1)

print(order_1)
print(order_2)
# print(order_3)
print('\n')
book.process_order(order_1)

print(book)

book.process_order(order_2)
book.process_order(order_3)

print(order_1)
print(order_2)
# print(order_3)

print(book)

print(account_1)
print(account_2)
print(book.get_bids())
print(book.get_asks())
# print("\n\nWHAAAAA\n\n")
# print(account_1)
# print(account_2)
# print(book.get_best_bid())
# print(book.get_best_ask())