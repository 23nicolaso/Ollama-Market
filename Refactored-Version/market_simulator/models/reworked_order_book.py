from market_simulator.utils.market_utils import last_prices, accounts
from market_simulator.price_server import emit_trade_update
from sortedcontainers import SortedDict
from collections import deque
import math

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

        # sorted dictionaries of orders, maintaining price time priority
        # bids use custom key function to sort in prices descending order 
        # NOTE: do not need to sort based on time for time priority, as insertion order is maintained
        self.bids = SortedDict(lambda k: -k[0])
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
            price = key[0]  # Price is the first element in the composite key
            if price not in bid_dict:
                bid_dict[price] = 0
            bid_dict[price] += order.quantity
        
        # Process asks - group by price and sum quantities
        ask_dict = {}
        for key, order in self.asks.items():
            price = key[0]  # Price is the first element in the composite key
            if price not in ask_dict:
                ask_dict[price] = 0
            ask_dict[price] += order.quantity
        
        # Convert to formatted strings and sort
        bid_pairs = {str(price): quantity for price, quantity in bid_dict.items()}
        ask_pairs = {str(price): quantity for price, quantity in ask_dict.items()}
        
        return bid_pairs, ask_pairs
    
    def matchBooks(self):
        return # DOES NOTHING, HERE JUST SO THE BENCHMARK SCRIPT WORKS 

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
        filled_keys = deque()  # Deque to track keys to remove after iteration
        side = "buy" if direction == 1 else "sell"
        oppSide = "sell" if side == "buy" else "buy"
        
        if direction == "buy":
            book = self.asks
            urgentBook = self.urgentSells
            urgentQuantity = self.urgentSellQuantity
            bookSize = self.askSize
        else:
            book = self.bids
            urgentBook = self.urgentBuys
            urgentQuantity = self.urgentBuyQuantity
            bookSize = self.bidSize

        if orderType == "limit":
            fillPrice = price
        else:
            fillPrice = self.getLastPrice()

        # If urgent book quantity < quantity, fill all urgent book and reduce remaining_qty accordingly
        if urgentQuantity < remaining_quantity:  
            # fill all urgent orders at once (for efficient batching)
            for order in urgentBook:
                order.fillEntireOrder(self.asset, fillPrice)
                emit_trade_update(order.accountID, self.asset, oppSide, order.quantity, fillPrice)
                emit_trade_update(accountID, self.asset, side, order.quantity, fillPrice)

            urgentBook.clear() # O(1) operation
            remaining_quantity -= urgentQuantity
            total_fill_price += fillPrice*urgentQuantity
            urgentQuantity = 0

        else: # urgent book quantity > quantity, so order will be fully filled
            urgentQuantity -= remaining_quantity
            # fill until remaining_quantity reached or urgent book is empty
            while remaining_quantity > 0:
                if len(urgentBook)==0:
                    break
                urgentOrder = urgentBook.popleft() 
                if urgentOrder.quantity <= 0: # empty order, skip to next iteration
                    continue

                if urgentOrder.quantity >= remaining_quantity: # case where current urgent order is larger than or equal to remaining quantity
                    fill_qty = remaining_quantity
                    urgentOrder.fillQuantity(self.asset, fill_qty, fillPrice) # reduce urgent order quantity by remaining quantity
                    
                    # update urgent sell/buy quantity accordingly
                    if direction == "buy":
                        self.urgentSellQuantity -= fill_qty
                    else:
                        self.urgentBuyQuantity -= fill_qty

                    total_fill_price += fillPrice * fill_qty
                    remaining_quantity = 0

                    # replace with queuing, to batch this heavy operation
                    emit_trade_update(urgentOrder.accountID, self.asset, oppSide, fill_qty, fillPrice)
                    emit_trade_update(accountID, self.asset, side, fill_qty, fillPrice)

                    if urgentOrder.quantity > 0:
                        urgentBook.appendleft(urgentOrder) # put partially filled urgent order back into front of queue
                    return quantity, total_fill_price
                
                else: # case where urgent order is smaller than remaining quantity
                    fill_qty = urgentOrder.quantity
                    urgentOrder.fillEntireOrder(self.asset, fillPrice)

                    # adjust quantities accordingly
                    if direction == "buy":
                        self.urgentSellQuantity -= fill_qty
                    else:
                        self.urgentBuyQuantity -= fill_qty

                    remaining_quantity -= fill_qty
                    total_fill_price += fillPrice * fill_qty

                    # replace with queuing of this to batch the heavy operation
                    emit_trade_update(urgentOrder.accountID, self.asset, oppSide, fill_qty, fillPrice)
                    emit_trade_update(accountID, self.asset, side, fill_qty, fillPrice)

        # Iterate through book in ascending price order (best prices first)
        # iterate with irange on dictionary so only orders meeting price condition are used
        if orderType == "limit":
            if direction == "buy": # filling against asks
                it = book.irange(maximum=(price, float('inf'))) 
            else: # filling against bids
                it = book.irange(maximum=(price, float('inf')))
        else:
            it = book.irange()
        
        # if book size < size of the order, then fully fill as much of book as possible
        if remaining_quantity > bookSize:
            if bookSize == 0:
                return quantity - remaining_quantity, total_fill_price

            orders = [book[key] for key in it] # Create list of all orders to fill in O(k)
            
            if orderType == "market":
                book.clear() # O(1)
            else:
                # automate decision of removal method based on k/n
                # check which is more efficient with O(k + log(n))
                # then use either delete iteratively: O(k log n) or bisect right: O(n)
                n = len(book) # O(1) 
                keys_to_remove = list(it) # O(k + log(n))
                k = len(keys_to_remove) # O(k)
                try:
                    cond = k < n / math.log(n)
                except: # n is 0 or 1, use basic batch delete
                    cond = True

                if cond:  # Optimal threshold
                    # Use batch delete
                    for key in it:
                        del book[key]
                else:
                    # Use reassign with slicing
                    if direction == "buy":
                        index = book.bisect_right((price, float('inf')))
                        self.asks = SortedDict(self.asks.islice(index, None))  # O(n)
                    else:
                        index = book.bisect_right((price, float('inf')))
                        self.bids = SortedDict(self.bids.islice(index, None))  # O(n)


            fill_qty = 0
            fill_price = 0 
            for order in orders: # batch fill all orders O(k)
                emit_trade_update(order.accountID, self.asset, oppSide, order.quantity, order.price)
                fill_qty += order.quantity
                fill_price += order.price
                order.fillEntireOrder(self.asset, order.price)

            emit_trade_update(accountID, self.asset, side, fill_qty, fill_price*fill_qty)
            total_fill_price += fill_price * fill_qty
            remaining_quantity -= fill_qty

            # reset size counter
            if direction == "buy":
                self.askSize = 0
            else:
                self.bidSize = 0

        else: # book size > size of order and market type
            for key in it:
                if remaining_quantity <= 0:
                    break

                order = book[key]
                
                if order.quantity == 0:
                    filled_keys.append(key)
                    continue # order is empty, skip and remove
                    
                # Calculate how much we can fill from this order
                fill_amount = min(remaining_quantity, order.quantity)

                # Always fills at other order's limit price
                fill_price = order.price

                # Update running totals
                remaining_quantity -= fill_amount
                bookSize -= fill_amount
                total_fill_price += fill_price * fill_amount

                if direction == "buy":
                    self.askSize -= fill_amount
                else:
                    self.bidSize -= fill_amount
                
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
        
            # Remove by bisecting dictionary to right of last accessed node 
            for key in filled_keys:
                book.pop(key)
    
        return quantity - remaining_quantity, total_fill_price

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
            # Create composite key from price and order ID for strict ordering
            composite_key = (price, order.orderID)
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
        return last_prices[self.asset] 