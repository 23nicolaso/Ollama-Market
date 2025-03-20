"""
Through strategic use of batching, data structures, this order book improves upon the prior matching engine for some operations.
"""

from market_simulator.utils.market_utils import last_prices, accounts
from market_simulator.price_server import emit_trade_update
from sortedcontainers import SortedDict
from collections import deque

class Order:
    __slots__ = ['orderID', 'price', 'quantity', 'accountID', 'orderType', 'direction']

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

        self._cached_depth = None # for getNearbyDepth
        self._depth_price = None

        self._urgentBuyQuantity = 0
        self._urgentSellQuantity = 0
        self._bidSize = 0
        self._askSize = 0
        self._best_bid = None
        self._best_ask = None
        self._mid_price = None
        self._last_price = None

        last_prices[asset] = initialPrice

    # Add method to invalidate cache when orderbook changes
    def _invalidate_cache(self):
        """Invalidate cached properties when orderbook state changes"""
        self._best_bid = None
        self._best_ask = None
        self._mid_price = None
        self._last_price = None

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
        bid_rng = self.lastPrice-depth
        ask_rng = self.lastPrice+depth
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
        
        print(f"\nUrgent Buys: {len(self.urgentBuys)} (Total Qty: {self._urgentBuyQuantity})")
        print(f"Urgent Sells: {len(self.urgentSells)} (Total Qty: {self._urgentSellQuantity})")
        print(f"Bid Size: {self._bidSize}, Ask Size: {self._askSize}")
        print(f"Last Price: {last_prices[self.asset]}")
        print("-----------------------------------\n")

    def _update_price_level(self, price, quantity, is_bid):
        """Update the price level cache for a given price"""
        cache = self._bid_levels if is_bid else self._ask_levels
        if quantity == 0:
            cache.pop(price, None)
        else:
            cache[price] = quantity
        self._cached_depth = None  # Invalidate depth cache

    def _rebuild_price_levels(self):
        """Rebuild entire price level cache"""
        self._bid_levels.clear()
        self._ask_levels.clear()
        
        # Rebuild bid levels
        for key, order in self.bids.items():
            price = key[0]
            self._bid_levels[price] = self._bid_levels.get(price, 0) + order.quantity
            
        # Rebuild ask levels
        for key, order in self.asks.items():
            price = key[0]
            self._ask_levels[price] = self._ask_levels.get(price, 0) + order.quantity

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
        update_queue = deque() # Add update queue to batch trade updates
        side = "buy" if direction == 1 else "sell"
        oppSide = "sell" if side == "buy" else "buy"
        
        if direction == "buy":
            book = self.asks
            urgentBook = self.urgentSells
            urgentQuantity = self._urgentSellQuantity
            bookSize = self._askSize
        else:
            book = self.bids
            urgentBook = self.urgentBuys
            urgentQuantity = self._urgentBuyQuantity
            bookSize = self._bidSize


        if orderType == "limit":
            fillPrice = price
        else:
            fillPrice = self.lastPrice

        if urgentQuantity > 0:
            # If urgent book quantity < quantity, fill all urgent book and reduce remaining_qty accordingly
            if urgentQuantity < remaining_quantity:  
                # fill all urgent orders at once (for efficient batching)
                for order in urgentBook:
                    order.fillEntireOrder(self.asset, fillPrice)
                    update_queue.append((order.accountID, self.asset, oppSide, order.quantity, fillPrice))
                    update_queue.append((accountID, self.asset, side, order.quantity, fillPrice))

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
                            self._urgentSellQuantity -= fill_qty
                        else:
                            self._urgentBuyQuantity -= fill_qty

                        total_fill_price += fillPrice * fill_qty
                        remaining_quantity = 0

                        # queue to batch this heavy operation
                        update_queue.append((urgentOrder.accountID, self.asset, oppSide, fill_qty, fillPrice))
                        update_queue.append((accountID, self.asset, side, fill_qty, fillPrice))

                        if urgentOrder.quantity > 0:
                            urgentBook.appendleft(urgentOrder) # put partially filled urgent order back into front of queue
                        return quantity, total_fill_price
                    
                    else: # case where urgent order is smaller than remaining quantity
                        fill_qty = urgentOrder.quantity
                        urgentOrder.fillEntireOrder(self.asset, fillPrice)

                        # adjust quantities accordingly
                        if direction == "buy":
                            self._urgentSellQuantity -= fill_qty
                        else:
                            self._urgentBuyQuantity -= fill_qty

                        remaining_quantity -= fill_qty
                        total_fill_price += fillPrice * fill_qty

                        # replace with queuing of this to batch the heavy operation
                        update_queue.append((urgentOrder.accountID, self.asset, oppSide, fill_qty, fillPrice))
                        update_queue.append((accountID, self.asset, side, fill_qty, fillPrice))

        # Iterate through book in ascending price order (best prices first)
        # iterate with irange on dictionary so only orders meeting price condition are used
        if orderType == "limit":
            if direction == "buy": # filling against asks
                it = book.irange(maximum=(price, float('inf'))) 
            else: # filling against bids
                it = book.irange(maximum=(price, float('inf')))
        else:
            it = book
        
        # if book size < size of the order, then fully fill as much of book as possible
        if remaining_quantity > bookSize:
            if bookSize == 0:
                # Process all queued updates before returning
                while update_queue:
                    acc_id, asset, side, qty, price = update_queue.popleft()
                    emit_trade_update(acc_id, asset, side, qty)
                return quantity - remaining_quantity, total_fill_price
            orders = [book[key] for key in it] # Create list of all orders to fill in O(k)
            
            if orderType == "market":
                book.clear() # O(1)
            else:
                # automatically select deletion method
                n = len(book) # O(1) 
                k = len(list(it)) # O(k)

                try:
                    cond = k < n / 3 # approx
                except:
                    cond = True
            
                if cond:  # Optimal threshold
                    # Use batch delete
                    for key in it:
                        del book[key]
                else: # Otherwise, more efficient to split and wipe
                    if direction == "buy":
                        # Keep only keys strictly greater than last key 
                        self.asks = SortedDict(
                            (k, self.asks[k]) for k in self.asks.irange(minimum=(price, float('inf')), maximum=None, inclusive=(True, False))
                        )
                    else:
                        # Bids use descending order, so we need keys > last_accessed_key
                        self.bids = SortedDict(
                            (k, self.bids[k]) for k in self.bids.irange(minimum=(price, float('inf')), maximum=None, inclusive=(True, False))
                        ).__class__(lambda k: -k[0])  # Reapply custom sorting
                
            fill_qty = 0
            fill_price = 0 
            for order in orders: # batch fill all orders O(k)
                update_queue.append((order.accountID, self.asset, oppSide, order.quantity, order.price))
                fill_qty += order.quantity
                fill_price += order.price * order.quantity
                order.fillEntireOrder(self.asset, order.price)

            update_queue.append((accountID, self.asset, side, fill_qty, fill_price))
            total_fill_price += fill_price
            remaining_quantity -= fill_qty

            # reset size counter
            if direction == "buy":
                self._askSize = 0
            else:
                self._bidSize = 0

        else: # book size >= size of order, fill until q is filled 
            last_filled = False
            last_accessed_key = None

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
                    self._askSize -= fill_amount
                else:
                    self._bidSize -= fill_amount
                
                # Update the order quantity or mark for removal
                if fill_amount == order.quantity:
                    filled_keys.append(key)  # Order completely filled, mark for removal
                    order.fillEntireOrder(self.asset, fill_price)
                    last_filled = True
                else:
                    order.fillQuantity(self.asset, fill_amount, fill_price) # Partially filled, update quantity
                    # Update last price for the asset
                    last_filled = False
                last_prices[self.asset] = fill_price
                
                # Emit trade update
                last_accessed_key = key
                update_queue.append((order.accountID, self.asset, oppSide, fill_amount, order.price))
                update_queue.append((accountID, self.asset, side, fill_amount, order.price))
        
            # Automatically select optimal method to use (either bisect or iteratively pop)
            if filled_keys and last_accessed_key:
                n = len(book) # O(1) 
                k = len(filled_keys) # O(k)

                try:
                    cond = k < n / 3 # approx
                except:
                    cond = True
            
                if cond:  # Optimal threshold
                    # Use batch delete
                    for key in filled_keys:
                        del book[key]
                else:
                    if direction == "buy":
                        # Keep only keys strictly greater than last_accessed_key
                        self.asks = SortedDict(
                            (k, self.asks[k]) for k in self.asks.irange(last_accessed_key, None, inclusive=(not last_filled, False))
                        )
                    else:
                        # Bids use descending order, so we need keys > last_accessed_key
                        self.bids = SortedDict(lambda k: -k[0], 
                            ((k, self.bids[k]) for k in self.bids.irange(last_accessed_key, None, inclusive=(not last_filled, False)))
                        )

        # Process all queued updates before returning
        while update_queue:
            acc_id, asset, side, qty, price = update_queue.popleft()
            emit_trade_update(acc_id, asset, side, qty)

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
        self._invalidate_cache()
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
                self._urgentBuyQuantity += quantity-qFilled
                # Ensure non-negative
                self._urgentBuyQuantity = max(0, self._urgentBuyQuantity)
            else:
                self._urgentSellQuantity += quantity-qFilled
                # Ensure non-negative
                self._urgentSellQuantity = max(0, self._urgentSellQuantity)
        else:
            if direction == "buy":
                self._bidSize += quantity-qFilled
            else:
                self._askSize += quantity-qFilled
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
            self._invalidate_cache()
            order = self.bids.pop(composite_key)
            self._bidSize -= order.quantity
            return True
        elif composite_key in self.asks:
            self._invalidate_cache()
            order = self.asks.pop(composite_key)
            self._askSize -= order.quantity
            return True
        return False

    def getUrgentQuantity(self):
        # returns sum of quantities of urgent orders
        return self._urgentBuyQuantity, self._urgentSellQuantity

    def getUrgentOrders(self):
        return self.urgentBuys, self.urgentSells

    def isOrderNone(self, composite_key):
        return composite_key not in self.bids and composite_key not in self.asks

    def cancelAllOldOrders(self):
        self.bids.clear()
        self._bidSize = 0
        self.asks.clear()
        self._askSize = 0

    def cancelOrdersByAccount(self, accountID):
        # slightly inefficient, preferable to use cancelOrder method
        for key in self.bids.keys():
            if self.bids[key].accountID == accountID:
                order = self.bids.pop(key)
                self._bidSize -= order.quantity
        for key in self.asks.keys():
            if self.asks[key].accountID == accountID:
                order = self.asks.pop(key)
                self._askSize -= order.quantity

    def display(self):
        print(f"{self.asset} Orderbook")
        print(f"Last Price: {self.lastPrice}")
        print(f"Bids, Asks: {self.getBidAskPairs()}")
        if self._urgentBuyQuantity>0:
            print(f"Urgent Buy Quantity: {self._urgentBuyQuantity}")
        if self._urgentSellQuantity> 0 :
            print(f"Urgent Sell Quantity: {self._urgentSellQuantity}")

    def displayPrice(self):
        print(f"{self.asset} price: {round(self.lastPrice, 2)}")

    def get_bidSize(self):
        return self._bidSize 

    def get_askSize(self):
        return self._askSize

    @property
    def bestBid(self):
        """Cached property for best bid price"""
        if not self.bids:
            return None
        if self._best_bid is None:
            self._best_bid = self.bids.keys()[0][0]
        return self._best_bid

    @property
    def bestAsk(self):
        """Cached property for best ask price"""
        if not self.asks:
            return None
        if self._best_ask is None:
            self._best_ask = self.asks.keys()[0][0]
        return self._best_ask

    def matchBooks(self):
        return # Doesn't do anything since orders are auto-matched

    def getUnfilledUrgentOrders(self):
        return max(0, self._urgentBuyQuantity), max(0, self._urgentSellQuantity)

    def fillUrgentOrders(self):
        return # Doesn't do anything since orders are auto-matched

    @property
    def lastPrice(self):
        """Cached property for last price"""
        if not last_prices[self.asset]:
            return None
        if self._last_price is None:
            self._last_price = last_prices[self.asset]
        return self._last_price