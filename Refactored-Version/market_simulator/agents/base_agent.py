from market_simulator.models.account import Account
from market_simulator.utils.market_utils import accounts

class MarketAgent:
    def __init__(self, accountID, cash):
        self.account = Account(accountID, cash)
        accounts[accountID] = self.account
        self.orders = []
        
    def placeOrder(self, orderBook, direction, price, quantity, orderType):
        order_key = orderBook.addOrder(direction, price, quantity, orderType, self.account.accountID)
        if order_key != 0:
            self.orders.append(order_key)
        return order_key

    def cancelAllOrders(self, orderBook):
        orderBook.cancelOrdersByAccount(self.account.accountID)

    def cancelOrder(self, orderBook, order):
        orderBook.cancelOrder(order)
        self.orders.remove(order)

    def checkOrders(self, orderBook):
        for order in self.orders:
            if orderBook.isOrderNone(order):
                self.orders.remove(order)

    def displayAccount(self):
        print(f"Account {self.account.accountID} has {self.account.getCash()} cash and the following positions: {self.account.positions}") 