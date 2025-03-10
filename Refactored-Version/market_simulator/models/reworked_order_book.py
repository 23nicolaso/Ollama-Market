'''
This is a reworked version of the order book that uses a sorted dictionary to store orders.
While it was designed to have far more efficient runtimes asymptotically, it is actually slower in practice.
This is due to the overhead caused by having a sorted dictionary with custom key functions, 
and because the original order book took advantage of efficient batching, whereas this model
uses too many individual operations, which is less efficient.

Here is a comparison of the asymptotic runtimes of each major operation in the two models: 
Operation       | Original Runtime | New Runtime
add order       | O(n)             | O(log(n))
get best prices | O(1)             | O(1)
matching books  | O(k*m)           | O(k*log(n))
cancelling      | O(n)             | O(log(n))
book maintenance| O(n)             | no maintenance needed
'''

from market_simulator.utils.market_utils import last_prices, accounts
from market_simulator.price_server import emit_trade_update
from market_simulator.portfolio_utils import calculate_portfolio_status, emit_portfolio_update
from uuid import uuid4
from sortedcontainers import SortedDict
from time import time

class Order:
    def __init__(self, direction, price, quantity, orderType, accountID):
        self.orderID = uuid4()
        self.price = price
        self.quantity = quantity
        self.accountID = accountID
        self.orderType = orderType
        self.direction = direction

    def fillQuantity(self, quantity):
        accounts[self.accountID].addPosition(self.asset, quantity*self.direction)
        accounts[self.accountID].addPosition("CASH", -self.price*quantity*self.direction)
        self.quantity -= quantity

    def fillEntireOrder(self):
        self.fillQuantity(self.quantity)
        self.quantity = 0


class OrderBook:
    def __init__(self, asset, initialPrice):
        self.asset = asset

        # sorted dictionaries of orders, sorted by price time priority
        # NOTE IN ORDER OF (orderType, price, time, orderID)

        # bids use custom key function to sort in prices descending order 
        self.bids = SortedDict(lambda k: (1 if k[0] == "limit" else 0, -k[1], k[2], k[3]))
        # asks use different custom key function
        self.asks = SortedDict(lambda k: (1 if k[0] == "limit" else 0, k[1], k[2], k[3]))

        last_prices[asset] = initialPrice

    def clearAgainstBook(self, quantity, direction, price, orderType, accountID):
        """Match buy order against available asks up to the specified quantity"""
        remaining_quantity = quantity
        total_fill_price = 0
        filled_keys = []  # Track keys to remove after iteration
        
        if direction == "buy":
            book = self.asks
        else:
            book = self.bids

        # Iterate through book in ascending price order (best prices first)
        for key, order in book.items():
            if remaining_quantity <= 0:
                break
            
            if order.quantity == 0:
                filled_keys.append(key)
                continue # order is empty, skip and remove
            
            # Break if order cannot be filled at price
            if orderType == "limit": 
                if direction == "buy" and order.price > price:
                    break
                elif direction == "sell" and order.price < price:
                    break
                
            # Calculate how much we can fill from this order
            fill_amount = min(remaining_quantity, order.quantity)

            if order.orderType == "market" and price is None:
                fill_price = last_prices[self.asset] * fill_amount
            elif order.orderType == "market" and price is not None:
                fill_price = price * fill_amount
            else:
                fill_price = order.price * fill_amount
            
            # Update running totals
            remaining_quantity -= fill_amount
            total_fill_price += fill_price
            
            # Update the order quantity or mark for removal
            if fill_amount == order.quantity:
                filled_keys.append(key)  # Order completely filled, mark for removal
            else:
                order.quantity -= fill_amount  # Partially filled, update quantity
                
            # Update last price for the asset
            last_prices[self.asset] = order.price
            
            # Emit trade update
            emit_trade_update(order.accountID, self.asset, order.direction, fill_amount, order.price)
            emit_trade_update(accountID, self.asset, order.direction, fill_amount, order.price)
            emit_portfolio_update(order.accountID, calculate_portfolio_status(accounts[order.accountID]))
            emit_portfolio_update(accountID, calculate_portfolio_status(accounts[accountID]))
        
        # Remove filled orders after iteration
        for key in filled_keys:
            book.pop(key)
        
        quantity_filled = quantity - remaining_quantity
        return quantity_filled, total_fill_price

    def addOrder(self, direction, price, quantity, orderType, accountID):
        # Handle incorrectly formatted orders
        if orderType not in ["limit", "market"]:
            raise ValueError("Invalid order type submitted: " + str(orderType))
        if direction not in ["buy", "sell"]:
            raise ValueError("Invalid direction submitted: " + str(direction))
        if quantity <= 0:
            raise ValueError("Quantity must be positive: " + str(quantity))
        if accountID not in accounts:
            raise ValueError("Account ID not found: " + str(accountID))
        if price < 0:
            print("Price must be positive: " + str(price))
            return
        
        # Fill as much as possible against other side of the book
        qFilled, totalFillPrice = self.clearAgainstBook(quantity, direction, price, orderType, accountID)
        
        translatedDirection = 1 if direction == "buy" else -1
        # Settle transacted quantity at total fill price
        accounts[accountID].addPosition(self.asset, qFilled*translatedDirection)
        accounts[accountID].addPosition("CASH", - totalFillPrice*translatedDirection)

        # Handle the case where the order is completely filled
        if qFilled == quantity:
            return 0
        
        # Otherwise, remaining quantity, so add an order to the order book
        # Create order object
        order = Order(direction, price, quantity-qFilled, orderType, accountID)

        # Create composite key from price, timestamp and order ID for strict ordering
        composite_key = (orderType, price, time(), order.orderID)

        # Add order to appropriate order book
        if direction == "buy":
            self.bids[composite_key] = order
        else:
            self.asks[composite_key] = order

        return composite_key
        
    def cancelOrder(self, composite_key):
        self.bids.pop(composite_key, None)
        self.asks.pop(composite_key, None)

    def getBids(self):
        return self.bids

    def getAsks(self):
        return self.asks
    
    def getMidPrice(self):
        return (self.bids.keys()[0][1] + self.asks.keys()[0][1]) / 2

    def getUrgentQuantity(self):
        # returns sum of quantities of urgent orders
        return sum([order.quantity for order in self.bids.values() if order.orderType == "market"]), sum([order.quantity for order in self.asks.values() if order.orderType == "market"])

    def getUrgentOrders(self):
        return [order.quantity for order in self.bids.values() if order.orderType == "market"], [order.quantity for order in self.asks.values() if order.orderType == "market"]

    def isOrderNone(self, order):
        return order not in self.bids.values() and order not in self.asks.values()

    def cancelAllOldOrders(self):
        self.bids.clear()
        self.asks.clear()

    def cancelOrdersByAccount(self, accountID):
        # slightly inefficient, preferable to use cancelOrder method
        for key in self.bids.keys():
            if self.bids[key].accountID == accountID:
                self.bids.pop(key)
        for key in self.asks.keys():
            if self.asks[key].accountID == accountID:
                self.asks.pop(key)

    def display(self):
        print(f"{self.asset} Orderbook")
        print(f"Total Buy Orders: {len(self.bids)}")
        print(f"Total Sell Orders: {len(self.asks)}")
        print(f"Last Price: {self.getLastPrice()}")
        print(f"Bids: {[str(value.netQuantity) + ' at '  + str(value.price) for value in sorted(self.bids.values(), key=lambda x: x.price, reverse=True)]}")
        print(f"Asks: {[str(value.netQuantity) + ' at ' + str(value.price) for value in sorted(self.asks.values(), key=lambda x: x.price)]}")
        print(f"Bid Size: {self.getBidSize()}")
        print(f"Ask Size: {self.getAskSize()}")

    def displayPrice(self):
        print(f"{self.asset} price: {round(self.getLastPrice(), 2)}")

    def getBidSize(self):
        return sum([bid.quantity for bid in self.bids.values()])

    def getAskSize(self):
        return sum([ask.quantity for ask in self.asks.values()])

    def getBestBid(self):
        return self.bids.keys()[0][1]

    def getBestAsk(self):
        return self.asks.keys()[0][1]

    def matchBooks(self):
        return self.bids.keys()[0][1], self.asks.keys()[0][1]

    def getUnfilledUrgentOrders(self):
        return self.bids.values()[0], self.asks.values()[0]

    def fillUrgentOrders(self):
        return self.bids.values()[0], self.asks.values()[0]

    def getLastPrice(self):
        return last_prices.get(self.asset, None) 