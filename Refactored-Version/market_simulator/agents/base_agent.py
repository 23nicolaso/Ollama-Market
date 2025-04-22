from market_simulator.models.account import Account
from market_simulator.utils.market_utils import accounts
from market_simulator.models.sortedListOB import LimitOrder, MarketOrder

class MarketAgent:
    def __init__(self, accountID, cash):
        self.account = Account(accountID, cash)
        accounts[accountID] = self.account
        
    def placeOrder(self, orderBook, direction, price, quantity, orderType):
        if orderType == "limit":
            # acc_id: int, quantity: int, price: float, side: Side)
            side = 1 if direction == "buy" else -1
            orderBook.process_order(LimitOrder(self.account.accountID, quantity, price, side))
        elif orderType == "market":
            # acc_id: int, quantity: int, side: Side):
            side = 1 if direction == "buy" else -1
            orderBook.process_order(MarketOrder(self.account.accountID, quantity, side))

    def cancelAllOrders(self, orderBook):
        orderBook.cancel_orders_from_account(self.account.accountID)

    def cancelOrder(self, orderBook, order):
        orderBook.cancel_order(order)
        self.orders.remove(order)

    def getPosition(self, asset):
        return self.account.getPosition(asset)

    def checkOrders(self, orderBook):
        for order in self.orders:
            if orderBook.isOrderNone(order):
                self.orders.remove(order)

    def displayAccount(self):
        print(f"Account {self.account.accountID} has {self.account.getCash()} cash and the following positions: {self.account.positions}") 