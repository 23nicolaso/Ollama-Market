#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

"""
reworked_order_book.pyx
"`. .`"`. .`"`. .`"`. .`"`. .`"`. .`"`. .`"`. .`"`. .`"`. .`
   `     `     `     `     `     `     `     `     `     `
    This is my slightly more optimal matching engine!
   .     .     .     .     .     .     .     .     .     .
_.` `._.` `._.` `._.` `._.` `._.` `._.` `._.` `._.` `._.` `.

Through strategic use of batching, data structures, this order book aims to improve upon the prior matching engine design.
Although in practice, it seems its actually much slower LOL!
    - Nicolas Ollivier
"""

from market_simulator.utils.market_utils import last_prices, accounts
from market_simulator.utils.db_utils import db_manager
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

    def fillQuantity(self, asset, quantity, price, placerID):
        accounts[self.accountID].addPosition(asset, quantity*self.direction)
        accounts[self.accountID].addPosition("CASH", -price*quantity*self.direction)
        last_prices[asset] = price
        self.quantity -= quantity

    def fillEntireOrder(self, asset, price, placerID):
        self.fillQuantity(asset, self.quantity, price, placerID)
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

        self.filledKeys = deque() # deque to store filled keys marked for removal

    # Add method to invalidate cache when orderbook changes
    def _invalidate_cache(self):
        """Invalidate cached properties when orderbook state changes"""
        self._best_bid = None
        self._best_ask = None
        self._mid_price = None

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

    def removeFilledKeys(self, direction):
        """Removes keys marked in self.filledKeys from the appropriate book."""
        if not self.filledKeys:
            return # Nothing to remove

        if direction == "buy": # Incoming order was buy, filled against asks
            book_to_modify = self.asks
            book_name = "asks"
        else: # Incoming order was sell, filled against bids
            book_to_modify = self.bids
            book_name = "bids"

        keys_to_remove = list(self.filledKeys)
        self.filledKeys.clear() # Clear the instance deque immediately

        removed_count = 0
        not_found_count = 0
        for key in keys_to_remove:
            try:
                # Directly attempt deletion
                del book_to_modify[key]
                removed_count += 1
            except KeyError:
                # Key wasn't found. This indicates a potential logic error elsewhere,
                # as the key should have been present in the book it was matched against.
                print(f"Warning: Key {key} scheduled for removal not found in {book_name} book (triggered by incoming {direction} order).")
                not_found_count += 1
            except Exception as e:
                print(f"Error removing key {key} from {book_name} book (triggered by incoming {direction} order): {e}")

        # Optional: Add a check if removed_count != len(keys_to_remove) to flag inconsistencies.
        if not_found_count > 0:
             print(f"Warning: {not_found_count}/{len(keys_to_remove)} keys scheduled for removal were not found.")


    def clearAgainstMarketBook(self, quantity, price, direction, accountID):
        # =^..^=   =^..^=   =^..^=    =^..^=    =^..^=    =^..^=    =^..^= #
        #               FILLING AGAINST MARKET ORDER BOOKS                 #
        # =^..^=   =^..^=   =^..^=    =^..^=    =^..^=    =^..^=    =^..^= #
                                
        # --------------------------------------------#
        # 1. Select appropriate books to work against #  
        # --------------------------------------------#  
        if direction == "sell":
            urgentBook = self.urgentBuys
            urgentQuantity = self._urgentBuyQuantity
        else:
            urgentBook = self.urgentSells
            urgentQuantity = self._urgentSellQuantity 

        # ------------------------------------------#
        # 2. Actually fill the orders against books #
        # ------------------------------------------# 
        if urgentQuantity > 0: 
            # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #
            # Case 1. Urgent Book Quantity < Quantity             #
            # Handle by filling all urgent orders Simultaneously  #
            # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #
            if urgentQuantity < quantity:  
                for order in urgentBook:
                    order.fillEntireOrder(self.asset, price, accountID)

                urgentBook.clear() # O(1) operation

                # Update urgent quantities accordingly
                if direction == "buy":
                    self._urgentSellQuantity = 0
                else:
                    self._urgentBuyQuantity = 0
            
                return urgentQuantity # local copy
                
            # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #
            # Case 2. Urgent Book Quantity >= Quantity            #
            # Fill iteratively until quantity filled              #
            # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #
            else: 
                remaining_quantity = quantity
                # Update urgent quantities accordingly
                if direction == "buy":
                    self._urgentSellQuantity -= quantity
                else:
                    self._urgentBuyQuantity -= quantity
                
                # fill until remaining_quantity reached or urgent book is empty
                while remaining_quantity > 0:
                    # ----------------- #
                    # Handle edge cases #
                    # ----------------- #
                    if len(urgentBook)==0:
                        break

                    urgentOrder = urgentBook.popleft()
                    if urgentOrder.quantity <= 0: # empty order, skip to next iteration
                        continue

                    # ----------------------------------------------------------------- #
                    # Case 1: Selected Market Order is sufficient to fill remaining qty # 
                    # ----------------------------------------------------------------- #
                    if urgentOrder.quantity >= remaining_quantity: # case where current urgent order is larger than or equal to remaining quantity
                        fill_qty = remaining_quantity
                        urgentOrder.fillQuantity(self.asset, fill_qty, price, accountID) # reduce urgent order quantity by remaining quantity

                        if urgentOrder.quantity > 0:
                            urgentBook.appendleft(urgentOrder) # put partially filled urgent order back into front of queue
                        return quantity
                    
                    # ------------------------------------------------------------------- #
                    # Case 2: Selected Market Order is insufficient to fill remaining qty # 
                    # ------------------------------------------------------------------- #
                    else: # case where urgent order is smaller than remaining quantity
                        fill_qty = urgentOrder.quantity
                        urgentOrder.fillEntireOrder(self.asset, price, accountID)
                        remaining_quantity -= fill_qty

    def clearAgainstLimitBook(self, quantity, direction, price, orderType, accountID):
        # =^..^=   =^..^=   =^..^=    =^..^=    =^..^=    =^..^=    =^..^= #
        #                FILLING AGAINST LIMIT ORDER BOOKS                 #
        # =^..^=   =^..^=   =^..^=    =^..^=    =^..^=    =^..^=    =^..^= #

        # --------------------------------------------#
        # 1. Select appropriate books to work against #  
        # --------------------------------------------#  
        if direction == "buy":
            book = self.asks
            bookSize = self._askSize
        else:
            book = self.bids
            bookSize = self._bidSize

        opposite_side = "sell" if direction == "buy" else "buy" 

        # --------------------------------------------#
        # 2. Make iterators based on price conditions #  
        # --------------------------------------------#  
        if orderType == "limit":
            if direction == "buy": 
                iterator = book.irange(maximum=(price, float('inf'))) # using this gives orders up to price
            else: 
                iterator = book.irange(maximum=(price, float('inf')))

        # -----------------------------------#
        # 3. Actually fill against the books #  
        # -----------------------------------#  
        
        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #
        # Case 1. User Places a Market Order.                 #
        #       a. Fill all limit orders (o.qty>=b.qty)       #
        #       b. Fill until (o.qty < b.qty)                 #
        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #
        if orderType == "market":
            fill_price = 0

            # -------- #
            # Case 1.a #
            # -------- #
            if quantity >= bookSize:
                # Pre-emptively set book size 0
                if direction == "buy":
                    self._askSize = 0
                else:
                    self._bidSize = 0
                
                # Add all orders to updateQueue
                for order in book.values(): 
                    fill_price += order.price * order.quantity
                    order.fillEntireOrder(self.asset, order.price, accountID)

                # Clear book
                book.clear()
                return bookSize, fill_price # python ints are immutable !
            
            # -------- #
            # Case 1.b #
            # -------- #
            else:
                # Pre-emptively reduce book size by quantity to fill
                if direction == "buy":
                    self._askSize -= quantity
                else:
                    self._bidSize -= quantity

                remaining_qty = quantity
                for key in book:
                    order = book[key]
                    if remaining_qty <= 0:
                        break # Stop iteration if fully filled
                    
                    if remaining_qty > order.quantity:
                        self.filledKeys.append(key)
                        fill_price += order.price * order.quantity
                        remaining_qty -= order.quantity
                        order.fillEntireOrder(self.asset, order.price, accountID)
                    
                    else: # remaining quantity <= order quantity
                        if order.quantity == remaining_qty:
                            self.filledKeys.append(key)
                            order.fillEntireOrder(self.asset, order.price, accountID)
                        else:
                            order.fillQuantity(self.asset, remaining_qty, order.price, accountID)

                        fill_price += order.price * remaining_qty
                    
                        self.removeFilledKeys(direction)

                        return quantity, fill_price

        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #
        # Case 2. User Places a Limit Order.                  #
        #       Fill until cond is no longer met              #
        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #
        else: # Incoming order is a limit order
            fill_price = 0
            if bookSize == 0:
                return 0, 0 # empty book means 0 filled @ 0

            remaining_qty = quantity
            # Use set(iterator) to create a static list of keys to process,
            # preventing issues if the dictionary changes during iteration.
            keys_to_process = set(iterator)

            for key in keys_to_process:
                if key not in book:
                    continue

                order = book[key] # Get the order from the opposite book (bid or ask)

                if remaining_qty <= 0:
                    break # Stop processing if the incoming order is fully filled

                # Limit orders fill at the price of the resting order in the book
                fill_this_order_at = order.price

                if remaining_qty >= order.quantity:
                    # The current book order `order` will be fully filled by the incoming order.
                    fill_amount = order.quantity
                    self.filledKeys.append(key) # Mark this book order's key for removal *later*
                    fill_price += fill_amount * fill_this_order_at
                    remaining_qty -= fill_amount
                    # Update the state of the book order object itself (quantity becomes 0)
                    # Note: This doesn't remove it from the 'book' dict yet.
                    order.fillEntireOrder(self.asset, fill_this_order_at, accountID)

                else:
                    # The current book order `order` is larger than the remaining quantity needed.
                    # It will be partially filled, and the incoming order will be fully satisfied.
                    fill_amount = remaining_qty
                    # Update the state of the book order object (reduce its quantity)
                    order.fillQuantity(self.asset, fill_amount, fill_this_order_at, accountID)
                    
                    fill_price += fill_amount * fill_this_order_at
                    remaining_qty = 0
                    # Do NOT add key to filledKeys, as the book order still has quantity left.
                    # Since the incoming order is now filled (remaining_qty is 0), break the loop.
                    break

            # --- After iterating through all potential matches ---
            filled_qty = quantity - remaining_qty # Total quantity filled from the incoming order

            # Update the total size of the book we were matching against
            if direction == "buy": # Incoming buy matched against asks
                self._askSize -= filled_qty
                self._askSize = max(0, self._askSize) # Prevent negative size
            else: # Incoming sell matched against bids
                self._bidSize -= filled_qty
                self._bidSize = max(0, self._bidSize) # Prevent negative size

            # NOW, after the loop is complete, remove all the book orders
            # that were fully filled and whose keys were added to filledKeys.
            # Return the total quantity filled and the total value exchanged

            self.removeFilledKeys(direction)
            return filled_qty, fill_price

    def clearAgainstBook(self, quantity, direction, price, orderType, accountID):
        """
        ("`-''-/").___..--''"`-._ 
        `6_ 6  )   `-.  (     ).`-.__.`) 
        (_Y_.)'  ._   )  `._ `. ``-..-' 
        _..`--'_..-_/  /--'_.'
        ((((.-''  ((((.'  (((.-' 
        
        Congratulations for finding the secret sauce! 
        """

        """
        The logical flow for order clearing here is as follows:
        1. Clear against urgent orders -> 2. Clear against book -> 3. return quantity filled and total fill price
        Fill price determination logic:
        MARKET -> MARKET: fill at last price of exchange
        LIMIT -> MARKET: fill at the limit price
        MARKET -> LIMIT: fill at the limit price
        LIMIT -> LIMIT: fill at the limit price of the first limit order
        """
        if quantity <= 0:
            return 0, 0
        remaining_quantity = quantity
        total_fill_price = 0
        
        # Handle Market -> Market, Limit -> Market 
        market_fill_price = price if orderType == "limit" else self.lastPrice 
        qty_filled_market = self.clearAgainstMarketBook(remaining_quantity, market_fill_price, direction, accountID)

        if qty_filled_market:
            total_fill_price += qty_filled_market * market_fill_price
            remaining_quantity -= qty_filled_market

        # Handle Market -> Limit, Limit -> Market
        result = self.clearAgainstLimitBook(remaining_quantity, direction, price, orderType, accountID)
        if result:
            limit_fill_price = result[1]
            qty_filled_limit = result[0]

            total_fill_price += limit_fill_price
            remaining_quantity -= qty_filled_limit
        
        # Returns quantity filled, and the price it was filled at
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
        if not price and orderType == "limit":
            return
        if price < 0:
            return
        
        # round price to 2 decimal places
        price = int(price*100)/100
        quantity = int(quantity)
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
    
    def getRemainingQuantity(self, composite_key):
        if composite_key in self.bids:
            return self.bids[composite_key].quantity
        elif composite_key in self.asks:
            return self.asks[composite_key].quantity
        else:
            return -1

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
        print(f"Bidsize, Asksize: ", self._bidSize, self._askSize)
        if self._urgentBuyQuantity>0:
            print(f"Urgent Buy Quantity: {self._urgentBuyQuantity}")
        if self._urgentSellQuantity> 0 :
            print(f"Urgent Sell Quantity: {self._urgentSellQuantity}")

    def displayPrice(self):
        print(f"{self.asset} price: {int(self.lastPrice*100)/100}")

    def get_bidSize(self):
        return self._bidSize 

    def get_askSize(self):
        return self._askSize

    @property
    def bestBid(self):
        """Cached property for best bid price"""
        if not self.bids:
            return self.lastPrice
        if self._best_bid is None:
            self._best_bid = self.bids.keys()[0][0]
        return self._best_bid

    @property
    def bestAsk(self):
        """Cached property for best ask price"""
        if not self.asks:
            return self.lastPrice
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