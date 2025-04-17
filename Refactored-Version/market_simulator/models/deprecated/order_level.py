import time
from market_simulator.utils.market_utils import last_prices, accounts

class OrderLevel:
    def __init__(self, price, quantity, asset, direction):
        self.asset = asset
        self.direction = 1 if direction == 1 else -1
        self.price = price
        self.netQuantity = quantity
        self.orders = []

    def __eq__(self, other):
        # other is an int here
        if isinstance(other, int):
            return self.netQuantity == other
    
    def __lt__(self, other):
        if isinstance(other, int):
            return self.netQuantity < other

    def __le__(self, other):
        if isinstance(other, int):
            return self.netQuantity <= other
    
    def __ge__(self, other):
        if isinstance(other, int):
            return self.netQuantity >= other
    
    def __gt__(self, other):
        if isinstance(other, int):
            return self.netQuantity > other



    def addOrder(self, quantity, accountID):
        self.netQuantity += quantity
        creation_time = time.time()
        self.orders.append([quantity, accountID, creation_time])

    def isEmpty(self):
        return self.netQuantity == 0

    def getPrice(self):
        return self.price

    def getQuantity(self):
        return self.netQuantity

    def fulfillAll(self):
        while self.orders:
            order = self.orders.pop(0)
            quantity, accountID, creation_time = order
            accounts[accountID].tradeAtPrice(self.asset, self.price, quantity, self.direction)
        
        self.netQuantity = 0
        last_prices[self.asset] = self.price 

    def cancelOldOrders(self):
        current_time = time.time()
        self.orders = [order for order in self.orders if current_time - order[2] < 100]
        self.netQuantity = sum([order[0] for order in self.orders])

    def cancelAll(self):
        self.orders = []
        self.netQuantity = 0

    def fulfillQuantity(self, quantityToFill):
        if quantityToFill >= self.netQuantity:
            quantityToFill -= self.netQuantity
            self.fulfillAll()
            return quantityToFill
        else:
            while quantityToFill > 0 and self.orders:
                order = self.orders[0]
                accountToModify = accounts[order[1]]

                if quantityToFill >= order[0]:
                    # Full order needs to be fulfilled
                    accountToModify.tradeAtPrice(self.asset, self.price, order[0], self.direction)
                    quantityToFill -= order[0]
                    self.netQuantity -= order[0]
                    self.orders.pop(0)
                    last_prices[self.asset] = self.price
                else:
                    # Order gets partially fulfilled
                    accountToModify.tradeAtPrice(self.asset, self.price, quantityToFill, self.direction)
                    order[0] -= quantityToFill
                    self.netQuantity -= quantityToFill
                    quantityToFill = 0
                    last_prices[self.asset] = self.price
            
            last_prices[self.asset] = self.price
            return 0

    def cancelOrdersFromID(self, accountID):
        new_orders = []
        for order in self.orders:
            if order[1] == accountID:
                self.netQuantity -= order[0]
            else:
                new_orders.append(order)
        self.orders = new_orders
        self.netQuantity = sum([order[0] for order in self.orders]) 

    def getAccountID(self):
        """Get the account ID of the first order in this level"""
        if self.orders:
            return self.orders[0][1]
        return None 