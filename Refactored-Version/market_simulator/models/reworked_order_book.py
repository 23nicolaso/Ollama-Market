'''
This is a reworked version of the order book that uses a sorted dictionary to store orders.
While it was designed to have far more efficient runtimes asymptotically, it is actually slower in practice.
This is due to the overhead caused by having a sorted dictionary with custom key functions, 
and because the original order book took advantage of efficient batching, whereas this model
uses too many individual operations, which is less efficient.
'''

from market_simulator.utils.market_utils import last_prices, accounts
from market_simulator.price_server import emit_trade_update
from market_simulator.portfolio_utils import calculate_portfolio_status, emit_portfolio_update
from sortedcontainers import SortedDict
from collections import deque
from time import time

class Order:
    def __init__(self, counter, direction, price, quantity, orderType, accountID):
        self.orderID = counter
        self.price = price
        self.quantity = quantity
        self.accountID = accountID
        self.orderType = orderType
        self.direction = 1 if direction == "buy" else -1

    def fillQuantity(self, asset, quantity, price):
        accounts[self.accountID].addPosition(asset, quantity*self.direction)
        accounts[self.accountID].addPosition("CASH", -price*quantity*self.direction)
        self.quantity -= quantity

    def fillEntireOrder(self, asset, price):
        self.fillQuantity(asset, self.quantity, price)
        self.quantity = 0

class OrderBook:
    def __init__(self, asset, initialPrice):
        self.asset = asset
        self.counter = 0 # used for orderIDs

        # sorted dictionaries of orders, sorted by price time priority
        # NOTE IN ORDER OF (orderType, price, time, orderID)

        # bids use custom key function to sort in prices descending order 
        self.bids = SortedDict(lambda k: (-k[1], k[2], k[3]))
        # asks use default key function
        self.asks = SortedDict()
        self.urgentBuys = deque()
        self.urgentSells = deque()
        self.urgentBuyQuantity = 0
        self.urgentSellQuantity = 0
        self.bidSize = 0
        self.askSize = 0

        last_prices[asset] = initialPrice

    def getBidAskPairs(self):
        """
        Returns bids and asks as lists of 'price: quantity' pairs
        Bids are sorted from highest to lowest price
        Asks are sorted from lowest to highest price
        """
        # Process bids - group by price and sum quantities
        bid_dict = {}
        for key, order in self.bids.items():
            price = key[1]  # Price is the second element in the composite key
            if price not in bid_dict:
                bid_dict[price] = 0
            bid_dict[price] += order.quantity
        
        # Process asks - group by price and sum quantities
        ask_dict = {}
        for key, order in self.asks.items():
            price = key[1]  # Price is the second element in the composite key
            if price not in ask_dict:
                ask_dict[price] = 0
            ask_dict[price] += order.quantity
        
        # Convert to formatted strings and sort
        bid_pairs = {str(price): quantity for price, quantity in bid_dict.items()}
        ask_pairs = {str(price): quantity for price, quantity in ask_dict.items()}
        
        return bid_pairs, ask_pairs
    
    def getNearbyDepth(self, depth):
        # returns how big bid,ask are in +/- depth of last price
        bid_rng = self.getLastPrice()-depth
        ask_rng = self.getLastPrice()+depth
        bidQ = 0
        askQ = 0
        for key in self.bids:
            if self.bids[key].price >= bid_rng:
                bidQ += self.bids[key].quantity
        
        for key in self.asks:
            if self.asks[key].price <= ask_rng:
                askQ += self.asks[key].quantity
        
        return bidQ, askQ

    def debugListBidsAsks(self):
        """
        Debug method to print all bids and asks with their composite keys and values.
        This is useful for debugging the order book state.
        """
        print(f"\n--- DEBUG: Order Book for {self.asset} ---")
        
        print("\nBIDS:")
        if len(self.bids) == 0:
            print("  No bids")
        else:
            for key, order in self.bids.items():
                print(f"  Key: {key} | Order: {order.__dict__}")
        
        print("\nASKS:")
        if len(self.asks) == 0:
            print("  No asks")
        else:
            for key, order in self.asks.items():
                print(f"  Key: {key} | Order: {order.__dict__}")
        
        print(f"\nUrgent Buys: {len(self.urgentBuys)} (Total Qty: {self.urgentBuyQuantity})")
        print(f"Urgent Sells: {len(self.urgentSells)} (Total Qty: {self.urgentSellQuantity})")
        print(f"Bid Size: {self.bidSize}, Ask Size: {self.askSize}")
        print(f"Last Price: {last_prices[self.asset]}")
        print("-----------------------------------\n")

    def clearEmptyOrderlevels(self):
        return # redundant
    
    def clearFarOrders(self):
        # Get current price
        return

    def clearAgainstBook(self, quantity, direction, price, orderType, accountID):
        """Match buy order against available asks up to the specified quantity
        Flows as follows:
        1. Clear against urgent orders -> 2. Clear against book -> 3. return quantity filled and total fill price
        Pricing fill logic:
        MARKET -> MARKET: fill at last price
        LIMIT -> MARKET: fill at limit price
        MARKET -> LIMIT: fill at limit price
        LIMIT -> LIMIT: fill at limit price of the first limit order
        """
        remaining_quantity = quantity
        total_fill_price = 0
        filled_keys = []  # Track keys to remove after iteration
        side = "buy" if direction == 1 else "sell"
        oppSide = "sell" if side == "buy" else "buy"
        
        if direction == "buy":
            book = self.asks
            urgentBook = self.urgentSells
            bookSize = self.askSize
        else:
            book = self.bids
            urgentBook = self.urgentBuys
            bookSize = self.bidSize

        # Clear against urgent orders first
        while len(urgentBook) > 0:
            if remaining_quantity <= 0:
                break
            
            urgentOrder = urgentBook.popleft() 
            if urgentOrder.quantity <= 0: # empty order, skip to next iteration
                continue

            if orderType == "market":
                fillPrice = last_prices[self.asset]
            else:
                fillPrice = price

            if urgentOrder.quantity >= remaining_quantity: # case where current urgent order is larger than or equal to remaining quantity
                fill_qty = remaining_quantity
                urgentOrder.fillQuantity(self.asset, fill_qty, fillPrice) # reduce urgent order quantity by remaining quantity
                if direction == "buy":
                    self.urgentSellQuantity -= fill_qty
                else:
                    self.urgentBuyQuantity -= fill_qty
                total_fill_price += fillPrice * fill_qty
                remaining_quantity = 0

                emit_trade_update(urgentOrder.accountID, self.asset, oppSide, fill_qty, fillPrice)
                emit_trade_update(accountID, self.asset, side, fill_qty, fillPrice)
                emit_portfolio_update(urgentOrder.accountID, calculate_portfolio_status(accounts[urgentOrder.accountID]))
                emit_portfolio_update(accountID, calculate_portfolio_status(accounts[accountID]))

                if urgentOrder.quantity > 0:
                    urgentBook.appendleft(urgentOrder) # put partially filled urgent order back into front of queue
                break
            
            elif urgentOrder.quantity < remaining_quantity: # case where urgent order is smaller than remaining quantity
                fill_qty = urgentOrder.quantity
                urgentOrder.fillEntireOrder(self.asset, fillPrice)
                if direction == "buy":
                    self.urgentSellQuantity -= fill_qty
                else:
                    self.urgentBuyQuantity -= fill_qty
                remaining_quantity -= fill_qty
                total_fill_price += fillPrice * fill_qty

                emit_trade_update(urgentOrder.accountID, self.asset, oppSide, fill_qty, fillPrice)
                emit_trade_update(accountID, self.asset, side, fill_qty, fillPrice)
                emit_portfolio_update(urgentOrder.accountID, calculate_portfolio_status(accounts[urgentOrder.accountID]))
                emit_portfolio_update(accountID, calculate_portfolio_status(accounts[accountID]))

        # Iterate through book in ascending price order (best prices first)
        for key, order in book.items():
            if remaining_quantity <= 0:
                break
            
            if order.quantity == 0:
                filled_keys.append(key)
                continue # order is empty, skip and remove
            
            # Break if order no orders match price condition
            if orderType == "limit": 
                if direction == "buy" and order.price > price:
                    break
                elif direction == "sell" and order.price < price:
                    break
                
            # Calculate how much we can fill from this order
            fill_amount = min(remaining_quantity, order.quantity)

            # Always fills at other order's limit price
            fill_price = order.price

            # Update running totals
            remaining_quantity -= fill_amount
            bookSize -= fill_amount
            total_fill_price += fill_price * fill_amount

            if direction == "buy":
                self.askSize = bookSize
            else:
                self.bidSize = bookSize
            
            # Update the order quantity or mark for removal
            if fill_amount == order.quantity:
                filled_keys.append(key)  # Order completely filled, mark for removal
                order.fillEntireOrder(self.asset, fill_price)
            else:
                order.fillQuantity(self.asset, fill_amount, fill_price) # Partially filled, update quantity
                # Update last price for the asset
            last_prices[self.asset] = order.price
            
            # Emit trade update
            emit_trade_update(order.accountID, self.asset, oppSide, fill_amount, order.price)
            emit_trade_update(accountID, self.asset, side, fill_amount, order.price)
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
        
        # round price to 2 decimal places
        price = round(price, 2)
        
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
        order = Order(self.counter, direction, price, quantity-qFilled, orderType, accountID)
        
        # Adjust quantity accordingly if market order
        if orderType == "market":
            if direction == "buy":
                self.urgentBuyQuantity += quantity-qFilled
                # Ensure non-negative
                self.urgentBuyQuantity = max(0, self.urgentBuyQuantity)
            else:
                self.urgentSellQuantity += quantity-qFilled
                # Ensure non-negative
                self.urgentSellQuantity = max(0, self.urgentSellQuantity)
        else:
            if direction == "buy":
                self.bidSize += quantity-qFilled
            else:
                self.askSize += quantity-qFilled
        self.counter += 1

        # Add order to appropriate order book
        if orderType == "limit":
            # Create composite key from price, timestamp and order ID for strict ordering
            composite_key = (orderType, price, time(), order.orderID)
            if direction == "buy":
                self.bids[composite_key] = order
            else:
                self.asks[composite_key] = order
        else:
            if direction == "buy":
                self.urgentBuys.append(order)
                return 0 # market orders can't be cancelled
            else:
                self.urgentSells.append(order)
                return 0 # market orders can't be cancelled

        return composite_key
        
    def cancelOrder(self, composite_key):
        if composite_key in self.bids:
            order = self.bids.pop(composite_key)
            self.bidSize -= order.quantity
            return True
        elif composite_key in self.asks:
            order = self.asks.pop(composite_key)
            self.askSize -= order.quantity
            return True
        return False

    def getBids(self):
        if not self.bids:
            return None
        return self.bids

    def getAsks(self):
        if not self.asks:
            return None
        return self.asks
    
    def getMidPrice(self):
        if not self.bids or not self.asks:
            return last_prices[self.asset]
        return (self.bids.keys()[0][1] + self.asks.keys()[0][1]) / 2

    def getUrgentQuantity(self):
        # returns sum of quantities of urgent orders
        return self.urgentBuyQuantity, self.urgentSellQuantity

    def getUrgentOrders(self):
        return self.urgentBuys, self.urgentSells

    def isOrderNone(self, composite_key):
        return composite_key not in self.bids and composite_key not in self.asks

    def cancelAllOldOrders(self):
        self.bids.clear()
        self.bidSize = 0
        self.asks.clear()
        self.askSize = 0

    def cancelOrdersByAccount(self, accountID):
        # slightly inefficient, preferable to use cancelOrder method
        for key in self.bids.keys():
            if self.bids[key].accountID == accountID:
                order = self.bids.pop(key)
                self.bidSize -= order.quantity
        for key in self.asks.keys():
            if self.asks[key].accountID == accountID:
                order = self.asks.pop(key)
                self.askSize -= order.quantity

    def display(self):
        print(f"{self.asset} Orderbook")
        print(f"Last Price: {self.getLastPrice()}")
        print(f"Bids, Asks: {self.getBidAskPairs()}")
        if self.urgentBuyQuantity>0:
            print(f"Urgent Buy Quantity: {self.urgentBuyQuantity}")
        if self.urgentSellQuantity> 0 :
            print(f"Urgent Sell Quantity: {self.urgentSellQuantity}")

    def displayPrice(self):
        print(f"{self.asset} price: {round(self.getLastPrice(), 2)}")

    def getBidSize(self):
        return self.bidSize 

    def getAskSize(self):
        return self.askSize

    def getBestBid(self):
        if not self.bids:
            return None
        return self.bids.keys()[0][1]

    def getBestAsk(self):
        if not self.asks:
            return None
        return self.asks.keys()[0][1]

    def matchBooks(self):
        return # Doesn't do anything since orders are auto-matched

    def getUnfilledUrgentOrders(self):
        return max(0, self.urgentBuyQuantity), max(0, self.urgentSellQuantity)

    def fillUrgentOrders(self):
        return # Doesn't do anything since orders are auto-matched

    def getLastPrice(self):
        return last_prices.get(self.asset, None) 