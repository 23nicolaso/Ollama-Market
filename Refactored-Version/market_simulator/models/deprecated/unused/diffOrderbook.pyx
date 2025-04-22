from pyskiplist import PySkipList

from market_simulator.utils.market_utils import last_prices, accounts
from market_simulator.price_server import emit_trade_update
from market_simulator.utils.db_utils import db_manager
from collections import deque
from dataclasses import dataclass

@dataclass(frozen=True)
class CompositeKey:
    trade_id: int
    price: float
    direction: int

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
        accounts[self.accountID].tradeAtPrice(asset, self.price, quantity, self.direction)
        last_prices[asset] = price
        self.quantity -= quantity

    def fillEntireOrder(self, asset, price):
        self.fillQuantity(asset, self.quantity, price)
        self.quantity = 0

class OrderLevel:
    def __init__(self, asset, price):
        self.asset = asset
        self.price = price
        self.quantity = 0
        self.orderDeque = deque()

    def addOrder(self, order):
        self.orderDeque.append(order)
        self.quantity += order.quantity
    
    def getOrderQuantity(self, key):
        if self.quantity == 0:
            return 0
        else:
            for order in self.orderDeque:
                if order.orderID == key:
                    return order.quantity
            return 0

    def removeOrder(self, key):
        if self.quantity == 0 :
            return 0
        else:
            for order in self.orderDeque:
                if order.orderID == key:
                    self.quantity -= order.quantity
                    self.orderDeque.remove(order)
                    return order.quantity
            return 0
                
    def removeOrdersFromAccount(self, accountID):
        if self.quantity == 0:
            return 0
        else:
            for order in list(self.orderDeque):
                if order.accountID == accountID:
                    self.quantity -= order.quantity
                    self.orderDeque.remove(order)
    
    def fillQuantity(self, quantity):
        if self.quantity == 0:
            return 0
        else:
            if quantity >= self.quantity:
                # run fill for all orders and wipe deque
                for order in self.orderDeque:
                    order.fillEntireOrder(self.asset, self.price)

                self.quantity = 0
                self.orderDeque.clear()
                return

            else:
                remainingQty = 0
                while len(self.orderDeque) > 0:
                    order = self.orderDeque.popleft()
                    if order.quantity == 0:
                        continue

                    if order.quantity > remainingQty:
                        order.fillQuantity(self.asset, remainingQty, self.price)
                        self.quantity -= remainingQty
                        self.orderDeque.appendleft(order)
                        return quantity
                    
                    else:
                        self.quantity -= order.quantity
                        remainingQty -= order.quantity
                        order.fillEntireOrder(self.asset, self.price)
                return quantity - remainingQty
    
    def fillEntireLevel(self):
        # run fill for all orders and wipe deque
        for order in self.orderDeque:
            order.fillEntireOrder(self.asset, self.price)

        returnQuantity = self.quantity
        self.quantity = 0
        self.orderDeque.clear()
        return returnQuantity


class OrderBook:
    def __init__(self, asset, initialPrice):
        self.asset = asset
        self.counter = 0 # used for orderIDs

        # use hybrid data structure - skiplist for checking prices, unsorted dict for storing orderlevels  
        self.bidPrices = PySkipList(False)
        self.askPrices = PySkipList(True)
        self.bidLevels = {}
        self.askLevels = {}
        self.urgentBuys = deque()
        self.urgentSells = deque()

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
        Returns aggregate of bids and asks as lists of 'price: quantity' pairs
        """
        # Process bids - group by price and sum quantities
        bid_dict = {}
        for priceKey in self.bidPrices:
            if priceKey not in self.bidLevels:
                continue
            else:
                bid_dict[str(priceKey)] = self.bidLevels[priceKey].quantity
        
        # Process asks - group by price and sum quantities
        ask_dict = {}
        for priceKey in self.askPrices:
            if priceKey not in self.askLevels:
                continue
            else:
                ask_dict[str(priceKey)] = self.askLevels[priceKey].quantity
        
        
        # Convert to formatted strings and sort
        bid_pairs = {str(price): quantity for price, quantity in bid_dict.items()}
        ask_pairs = {str(price): quantity for price, quantity in ask_dict.items()}

        return bid_pairs, ask_pairs

    def debugListBidsAsks(self):
        """
        Debug method to print all bids and asks with their composite keys and values.
        This is useful for debugging the order book state.
        """
        print(f"\n--- DEBUG: Order Book for {self.asset} ---")
        
        print("\nBIDS:")
        self.bidLevels.print()

        print("\nASKS:")
        self.askLevels.print()
        
        print(f"\nUrgent Buys: {len(self.urgentBuys)} (Total Qty: {self._urgentBuyQuantity})")
        print(f"Urgent Sells: {len(self.urgentSells)} (Total Qty: {self._urgentSellQuantity})")
        print(f"Bid Size: {self._bidSize}, Ask Size: {self._askSize}")
        print(f"Last Price: {last_prices[self.asset]}")
        print("-----------------------------------\n")

    def clearAgainstMarketBook(self, quantity, price, direction, accountID):
        # =^..^=   =^..^=   =^..^=    =^..^=    =^..^=    =^..^=    =^..^= #
        #               FILLING AGAINST MARKET ORDER BOOKS                 #
        # =^..^=   =^..^=   =^..^=    =^..^=    =^..^=    =^..^=    =^..^= #
                                
        # --------------------------------------------#
        # 1. Select appropriate books to work against #  
        # --------------------------------------------#  
        if direction == -1:
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
                    order.fillEntireOrder(self.asset, price)

                urgentBook.clear() # O(1) operation

                # Update urgent quantities accordingly
                if direction == 1:
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
                if direction == 1:
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
                        urgentOrder.fillQuantity(self.asset, fill_qty, price) # reduce urgent order quantity by remaining quantity
                        # queue to batch this heavy operation

                        if urgentOrder.quantity > 0:
                            urgentBook.appendleft(urgentOrder) # put partially filled urgent order back into front of queue
                        return quantity
                    
                    # ------------------------------------------------------------------- #
                    # Case 2: Selected Market Order is insufficient to fill remaining qty # 
                    # ------------------------------------------------------------------- #
                    else: # case where urgent order is smaller than remaining quantity
                        fill_qty = urgentOrder.quantity
                        urgentOrder.fillEntireOrder(self.asset, price)
                        remaining_quantity -= fill_qty

    def clearAgainstLimitBook(self, quantity, direction, price, orderType, accountID):
        # =^..^=   =^..^=   =^..^=    =^..^=    =^..^=    =^..^=    =^..^= #
        #                FILLING AGAINST LIMIT ORDER BOOKS                 #
        # =^..^=   =^..^=   =^..^=    =^..^=    =^..^=    =^..^=    =^..^= #

        # --------------------------------------------#
        # 1. Select appropriate books to work against #  
        # --------------------------------------------#  
        if direction == 1:
            bookPrices = self.askPrices
            orderMap = self.askLevels
            bookSize = self._askSize
            if orderType == False:
                price = float('inf')
        else:
            bookPrices = self.bidPrices
            orderMap = self.bidLevels
            bookSize = self._bidSize
            if orderType == False:
                price = 0

        # -------------------------------------------------#
        # 2. Get list of prices which meet price condition #  
        # -------------------------------------------------#  
        suitablePrices = bookPrices.get_up_to(price, inclusive=True)
        
        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #
        # Case 1. User Places a Market Order.                 #
        #       a. Fill all limit orders (o.qty>=b.qty)       #
        #       b. Fill until (o.qty < b.qty)                 #
        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #
        if orderType == False:
            # -------- #
            # Case 1.a #
            # -------- #
            fill_price = 0
            if quantity >= bookSize:
                # Pre-emptively set book size 0
                if direction == 1:
                    self._askSize = 0
                else:
                    self._bidSize = 0
                
                for price_level in suitablePrices:
                    if orderMap.get(price_level): 
                        qty = orderMap[price_level].fillEntireLevel()
                        fill_price += qty * price_level
                    else:
                        continue

                # Clear book
                bookPrices.clear()
                orderMap.clear()
                return bookSize, fill_price
            
            # -------- #
            # Case 1.b #
            # -------- #
            else:
                # Pre-emptively reduce book size by quantity to fill
                fill_price = 0
                if direction == 1:
                    self._askSize -= quantity
                else:
                    self._bidSize -= quantity

                remaining_qty = quantity

                for priceKey in suitablePrices:
                    if not orderMap.get(priceKey):
                        continue
                    orderLevel = orderMap[priceKey]

                    if remaining_qty > orderLevel.quantity:
                        fill_price += priceKey * orderLevel.quantity
                        remaining_qty -= orderLevel.quantity
                        orderLevel.fillEntireLevel()
                        del orderMap[priceKey]
                    
                    else: # remaining quantity <= order quantity
                        if orderLevel.quantity == remaining_qty:
                            orderLevel.fillEntireLevel()
                            del orderMap[priceKey]
                            bookPrices.erase_up_to(priceKey, inclusive = True)
                        else:
                            orderLevel.fillQuantity(remaining_qty)
                            bookPrices.erase_up_to(priceKey, inclusive = False)

                        fill_price += priceKey * remaining_qty
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

            for priceKey in suitablePrices:
                if remaining_qty <= 0:
                    break # Stop processing if the incoming order is fully filled

                if priceKey not in orderMap:
                    continue # Go onto next loop

                order = orderMap[priceKey] # Get the order from the opposite book (bid or ask)

                if remaining_qty >= order.quantity:
                    # The current book order `order` will be fully filled by the incoming order.
                    fill_price += order.quantity * priceKey
                    remaining_qty -= order.quantity
                    order.fillEntireLevel()
                    del orderMap[priceKey]
                    bookPrices.erase_up_to(priceKey, True)
                else:
                    # The current book order `order` is larger than the remaining quantity needed.
                    # It will be partially filled, and the incoming order will be fully satisfied.
                    fill_amount = remaining_qty
                    # Update the state of the book order object (reduce its quantity)
                    order.fillQuantity(fill_amount)
                    
                    fill_price += fill_amount * priceKey
                    remaining_qty = 0
                    # Since the incoming order is now filled (remaining_qty is 0), break the loop.
                    break

            # --- After iterating through all potential matches ---
            filled_qty = quantity - remaining_qty # Total quantity filled from the incoming order

            # Update the total size of the book we were matching against
            if direction == 1: # Incoming buy matched against asks
                self._askSize -= filled_qty
                self._askSize = max(0, self._askSize) # Prevent negative size
            else: # Incoming sell matched against bids
                self._bidSize -= filled_qty
                self._bidSize = max(0, self._bidSize) # Prevent negative size

            # NOW, after the loop is complete, remove all the book orders
            # that were fully filled and whose keys were added to filledKeys.
            # Return the total quantity filled and the total value exchanged

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
        remaining_quantity = quantity
        total_fill_price = 0
        
        # Handle Market -> Market, Limit -> Market 
        market_fill_price = price if orderType == True else self.lastPrice 
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
        if orderType not in [True, False]: # limit / market
            raise ValueError("Invalid order type submitted: " + str(orderType))
        if direction not in [1, -1]: # buy / sell
            raise ValueError("Invalid direction submitted: " + str(direction))
        if quantity <= 0:
            raise ValueError("Quantity must be positive: " + str(quantity))
        if accountID not in accounts:
            raise ValueError("Account ID not found: " + str(accountID))
        if not price and orderType == True:
            print("no price")
            return
        if price < 0:
            print("Price must be positive: " + str(price))
            return
        
        # round price to 2 decimal places
        price = round(price, 2)
        quantity = int(quantity)
        self._invalidate_cache()
        # Fill as much as possible against other side of the book
        qFilled, totalFillPrice = self.clearAgainstBook(quantity, direction, price, orderType, accountID)
        
        # Settle transacted quantity at total fill price
        accounts[accountID].addPosition(self.asset, qFilled*direction)
        accounts[accountID].addPosition("CASH", - totalFillPrice*direction)

        # Handle the case where the order is completely filled
        if qFilled == quantity:
            return 0
        
        # Otherwise, remaining quantity, so add an order to the order book
        # Create order object
        order = Order(self.counter, direction, price, quantity-qFilled, orderType, accountID)

        # Adjust quantity accordingly if market order
        if orderType == False:
            if direction == 1:
                self._urgentBuyQuantity += quantity-qFilled
                # Ensure non-negative
                self._urgentBuyQuantity = max(0, self._urgentBuyQuantity)
            else:
                self._urgentSellQuantity += quantity-qFilled
                # Ensure non-negative
                self._urgentSellQuantity = max(0, self._urgentSellQuantity)
        else:
            if direction == 1:
                self._bidSize += quantity-qFilled
            else:
                self._askSize += quantity-qFilled
        self.counter += 1

        # Add order to appropriate order book
        if orderType == True:
            # Create composite key from price and order ID for strict ordering
            composite_key = CompositeKey(order.orderID, price, direction)
            if direction == 1:
                if price in self.bidLevels:
                    self.bidLevels[price].addOrder(order)
                else:
                    self.bidLevels[price] = OrderLevel(self.asset, price)
                    self.bidLevels[price].addOrder(order)

                self.bidPrices.insert(price)
            else:
                if price in self.askLevels:
                    self.askLevels[price].addOrder(order)
                else:
                    self.askLevels[price] = OrderLevel(self.asset, price)
                    self.askLevels[price].addOrder(order)

                self.askPrices.insert(price)
                
        else:
            if direction == 1:
                self.urgentBuys.append(order)
                return 0 # market orders don't have an id 
            else:
                self.urgentSells.append(order)
                return 0 # market orders don't have an id

        return composite_key
        
    def cancelOrder(self, composite_key):
        if composite_key != 0:
            if composite_key.direction == 1:
                if self.bidLevels.get(composite_key.price):
                    q = self.bidLevels[composite_key.price].removeOrder(composite_key.trade_id)
                    self._bidSize -= q
                    if self.bidLevels[composite_key.price].quantity == 0:
                        self._invalidate_cache()
                        del self.bidLevels[composite_key.price]
                        self.bidPrices.remove(composite_key.price)
                    return True
            else:
                if self.askLevels.get(composite_key.price):
                    q = self.askLevels[composite_key.price].removeOrder(composite_key.trade_id)
                    self._askSize -= q
                    if self.askLevels[composite_key.price].quantity == 0:
                        self._invalidate_cache()
                        del self.askLevels[composite_key.price]
                        self.askPrices.remove(composite_key.price)
                    return True
        return False
    
    def getRemainingQuantity(self, composite_key):
        if composite_key.direction == 1:
            if self.bidLevels.get(composite_key):
                return self.bidLevels[composite_key.price].getOrderQuantity(composite_key.trade_id)
            return 0
        elif composite_key.direction == -1:
            if self.askLevels.get(composite_key):
                return self.askLevels[composite_key.price].getOrderQuantity(composite_key.trade_id)
            return 0
        else:
            return 0

    def getUrgentQuantity(self):
        # returns sum of quantities of urgent orders
        return self._urgentBuyQuantity, self._urgentSellQuantity

    def getUrgentOrders(self):
        return self.urgentBuys, self.urgentSells

    def isOrderNone(self, composite_key):
        if composite_key.direction == 1:
            if self.bidLevels.get(composite_key.price):
                return self.bidLevels[composite_key.price].getOrderQuantity(composite_key.trade_id) > 0
        else:
            if self.askLevels.get(composite_key.price):
                return self.askLevels[composite_key.price].getOrderQuantity(composite_key.trade_id) > 0
        return 0

    def cancelAllOldOrders(self):
        self.bidLevels.clear()
        self.bidPrices.clear()
        self._bidSize = 0
        self.askLevels.clear()
        self.askPrices.clear()
        self._askSize = 0

    def cancelOrdersByAccount(self, accountID):
        for level in self.bidLevels:
            level.removeOrdersFromAccount(accountID)

        for level in self.askLevels:
            level.removeOrdersFromAccount(accountID)

    def display(self):
        print(f"{self.asset} Orderbook")
        print(f"Last Price: {self.lastPrice}")
        print(f"Bids, Asks:")
        print([self.bidLevels[key].quantity for key in self.bidLevels])
        print([self.askLevels[key].quantity for key in self.askLevels])

        print("Bid prices, Ask prices: ")
        print(self.bidPrices)
        print(self.askPrices)
        print(f"Bidsize, Asksize: ", self._bidSize, self._askSize)
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
            return self.lastPrice
        if self._best_bid is None:
            self._best_bid = self.bidLevels.getFirst()
        return self._best_bid

    @property
    def bestAsk(self):
        """Cached property for best ask price"""
        if not self.asks:
            return self.lastPrice
        if self._best_ask is None:
            self._best_ask = self.askLevels.getFirst()
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