"""
"`. .`"`. .`"`. .`"`. .`"`. .`"`. .`"`. .`"`. .`"`. .`"`. .`
   `     `     `     `     `     `     `     `     `     `
    This is my last attempt at optimizing the design of
    my single-threaded matching engine... For now!
        - Nicolas Ollivier
   .     .     .     .     .     .     .     .     .     .
_.` `._.` `._.` `._.` `._.` `._.` `._.` `._.` `._.` `._.` `.

This matching engine improves on my previous iterations by 
taking full advantage of numpy to hopefully improve
performance! (actually much slower, don't use LOL) 
"""

import numpy as np
from market_simulator.utils.market_utils import last_prices, accounts
from market_simulator.price_server import emit_trade_update
from sortedcontainers import SortedDict
from collections import deque

class OrderArray:
    def __init__(self, initial_capacity=10000):
        # Create structured array with columns
        self.orders = np.zeros(initial_capacity, dtype=[
            ('direction', np.int8),     # 1 for buy, -1 for sell
            ('price', np.float32),      
            ('quantity', np.uint32),    
            ('accountID', 'U16'),       # Unicode string for account ID
            ('active', np.bool_)        # Mask for active orders
        ])
        self.capacity = initial_capacity
        # Initialize all orders as inactive
        self.orders['active'].fill(False)
        # Track number of active orders for resize decisions
        self.n_active = 0
        # Track next insertion point
        self.next_index = 0

    def get_order(self, key):
        # For a given key, returns all the information in order
        # direction, price, quantity, accountID
        if not self.orders[key]['active']:
            return 0
        
        # subtract quantity change
        return self.orders[key]['direction'], self.orders[key]['price'], self.orders[key]['quantity'], self.orders[key]['accountID']

    def insert(self, direction, price, quantity, accountID):
        # Check if we need more space
        if self.n_active >= self.capacity * 0.9:  # Resize at 90% capacity
            self._resize()
        
        # Find first inactive slot using numpy
        if self.next_index >= self.capacity:
            self.next_index = np.where(~self.orders['active'])[0][0]
        
        # Insert the order
        self.orders[self.next_index] = (
            1 if direction == "buy" else -1,
            price,
            quantity,
            accountID,
            True
        )
        
        slot = self.next_index
        self.n_active += 1
        self.next_index += 1
        return slot

    def modify_quantity(self, index, quantity_change):
        if not self.orders[index]['active']:
            return 0
        
        # subtract quantity change
        self.orders[index]['quantity'] -= quantity_change
        return self.orders[index]['quantity']

    def get_account_quantity(self, index):
        if not self.orders[index]['active']:
            return 0
        return self.orders[index]['accountID'], self.orders[index]['quantity']

    def get_quantity(self, index):
        if not self.orders[index]['active']:
            return 0
        return self.orders[index]['quantity']

    def get_quantity_price_direction(self, index):
        if not self.orders[index]['active']:
            return 0
        return self.orders[index]['quantity'], self.orders[index]['price'], self.orders[index]['direction']

    def remove(self, index):
        if self.orders[index]['active']:
            self.orders[index]['active'] = False
            self.n_active -= 1
            # If this index is before our next_index, update next_index
            if index < self.next_index:
                self.next_index = index

    def get_active_orders(self):
        """Return view of only active orders for vectorized operations"""
        return self.orders[self.orders['active']]

    def get_orders_by_account(self, accountID):
        """Efficiently get all indices corresponding with active orders for an account"""
        mask = (self.orders['active']) & (self.orders['accountID'] == accountID)
        return np.where(mask)[0] # returns array of indices where mask is true

    def deactive_by_filled_keys(self, filled_keys):
        """Efficiently deactive multiple orders using vectorized operations"""
        if not filled_keys.any():
            return
        
        keys_array = np.array(filled_keys)

        # Create mask and deactivate orders in one vectorized operation
        self.orders['active'][keys_array] = False
        
        # Update n_active count
        self.n_active -= len(filled_keys)
        
        # Update next_index if necessary
        if self.n_active > 0:
            min_deactivated = min(filled_keys)
            if self.next_index > min_deactivated:
                self.next_index = min_deactivated

    def set_orders_inactive_by_account(self, accountID):
        """Efficiently set all active orders for an account to inactive"""
        # Create mask for active orders belonging to the account
        mask = (self.orders['active']) & (self.orders['accountID'] == accountID)
        # Count how many orders we're deactivating
        orders_to_deactivate = np.sum(mask)
        # Set those orders to inactive
        self.orders['active'][mask] = False
        self.n_active -= orders_to_deactivate

        # Update next_index if necessary
        if orders_to_deactivate > 0:
            first_deactivated = np.where(mask)[0][0]
            if self.next_index > first_deactivated:
                self.next_index = first_deactivated

    def _resize(self):
        new_capacity = self.capacity * 2
        new_orders = np.zeros(new_capacity, dtype=self.orders.dtype)
        # Only copy active orders, automatically defragmenting the array
        active_mask = self.orders['active']
        new_orders[:self.n_active] = self.orders[active_mask]
        new_orders['active'][:self.n_active] = True

class OrderBook:
    class OrderLevel:
        __slots__ = ['_quantity', 'orders']
        def __init__(self, location, quantity):
            self._quantity = quantity
            self.orders = deque()
            self.orders.append(location) 

        def pushOrder(self, location, quantity):
            # add order to level
            self._quantity += quantity
            self.orders.append(location)

        def remove(self, location, quantity):
            try:
                self._quantity -= quantity
                self.orders.remove(location)
            except ValueError:
                print("Order to cancel not found")

        @property
        def quantity(self):
            return self._quantity


    def __init__(self, asset, initialPrice):
        self.asset = asset

        # the four books
        self.bids = SortedDict(lambda k: -k) # -k for descending sort
        self.asks = SortedDict()
        self.marketBuys = deque()
        self.marketSells = deque()
        self.orderArray = OrderArray()

        self.emissionQueue = deque()
        self.accountUpdateQueue = deque()
        self.accountCashUpdateQueue = deque()

        # value stores for fast access
        self._marketBuyQuantity = 0
        self._marketSellQuantity = 0
        self._bidSize = 0
        self._askSize = 0
        self._best_bid = None
        self._best_ask = None
        self._last_price = initialPrice

        # initiate last_prices with provided initial price
        last_prices[asset] = initialPrice

    def fillMarketKeys(self, keys, price):
        """
        Provided a list of keys and a price,
        Fills all the corresponding market orders
        Note : DOES NOT remove the keys from the market books, that is done elsewhere
        """
        if not keys:
            return

        # Get all order data in one vectorized operation
        keys_array = np.array(keys)
        mask = self.orderArray.orders['active'][keys_array]
        valid_keys = keys_array[mask]

        if len(valid_keys) == 0:
            return
        
        orders = self.orderArray.orders[valid_keys]
        dir_map = np.where(orders['direction'] == 1, "buy", "sell")

        quantities = orders['quantity']
        directions = orders['direction']

        for i in range(len(valid_keys)):
            self.emission_queue.append((orders['accountID'][i], dir_map[i], quantities[i]))
            self.account_update_queue.append((orders['accountID'][i], -directions[i]*quantities[i]))
            self.account_update_cash_queue.append((orders['accountID'][i], directions[i]*quantities[i]*price))

        # final step mark all inactive 
        self.orderArray.deactive_by_filled_keys(valid_keys)

    def fillLimitKeys(self, keys):
        """
        Provided a list of keys, 
        Fills all of corresponding limit orders 
        Note : DOES NOT remove the keys from the limit books
        """
        if not keys:
            return

        # Get all order data in one vectorized operation
        keys_array = np.array(keys)
        mask = self.orderArray.orders['active'][keys_array]
        valid_keys = keys_array[mask]

        if len(valid_keys) == 0:
            return
        
        orders = self.orderArray.orders[valid_keys]
        dir_map = np.where(orders['direction'] == 1, "buy", "sell")

        quantities = orders['quantity']
        directions = orders['direction']
        prices = orders['price']

        for i in range(len(valid_keys)):
            self.emissionQueue.append((orders['accountID'][i], dir_map[i], quantities[i]))
            self.accountUpdateQueue.append((orders['accountID'][i], directions[i]*quantities[i]))
            self.accountCashUpdateQueue.append((orders['accountID'][i], -directions[i]*quantities[i]*prices[i]))

        # final step mark all inactive 
        self.orderArray.deactive_by_filled_keys(valid_keys)

    def getBidAskPairs(self):
        """
        Returns bids and asks as lists of 'price: quantity' pairs
        """
        bid_pairs = {str(price): level.quantity for price, level in self.bids.items()}
        ask_pairs = {str(price): level.quantity for price, level in self.asks.items()}

        return bid_pairs, ask_pairs
    
    def matchBooks(self):
        return # DOES NOTHING, JUST HERE FOR THE BENCHMARK SCRIPT 

    def getNearbyDepth(self, depth):
        # returns how big bid,ask are in +/- depth of last price
        bid_rng = self.lastPrice-depth
        ask_rng = self.lastPrice+depth
        bidQ = 0
        askQ = 0
        for price, value in self.bids:
            if price >= bid_rng:
                bidQ += value[0] # here value[0] corresponds with Q @ p
        
        for price, value in self.asks:
            if price <= ask_rng:
                askQ += value[0]
        
        return bidQ, askQ
    
    def process_queued_updates(self):
        """Helper method to process all queued updates in batch"""
        while self.emissionQueue:
            acc_id, side, qty = self.emissionQueue.popleft()
            emit_trade_update(acc_id, self.asset, side, qty)

        while self.accountUpdateQueue:
            oppID, qty = self.accountUpdateQueue.popleft()
            accounts[oppID].addPosition(self.asset, qty)
        
        while self.accountCashUpdateQueue:
            oppID2, cash = self.accountCashUpdateQueue.popleft()
            accounts[oppID2].addPosition("CASH", cash)

    def process_limit_orders(self, book, it, remaining_quantity, direction, orderType, fill_price):
        """Helper method to process limit orders efficiently"""
        if remaining_quantity > self._askSize if direction == "buy" else self._bidSize:
            return self._process_full_book_fill(book, it, direction, orderType, fill_price)
        else:
            return self._process_partial_book_fill(book, it, remaining_quantity, direction, fill_price)

    def _process_full_book_fill(self, book, it, direction, orderType, fill_price):
        """Process case where order size > book size"""
        order_levels = list(it)
        all_orders = []
        total_quantity = 0
        total_fill_price = 0
        
        for level_price in order_levels:
            level = book[level_price]
            level_orders = list(level.orders)
            all_orders.extend(level_orders)
            total_quantity += level.quantity
            total_fill_price += level_price * level.quantity
        
        # Batch process all orders
        if all_orders:
            self.fillLimitKeys(all_orders)
        
        # Efficient book cleanup
        if orderType == "market":
            book.clear()
        else:
            # Use your existing efficient deletion method
            n, k = len(book), len(order_levels)
            if k < n / 5:
                for key in order_levels:
                    del book[key]
            else:
                self._rebuild_book_after_fill(book, direction, fill_price)
        
        return total_quantity, total_fill_price

    def _process_partial_book_fill(self, book, it, remaining_quantity, direction, fill_price):
        """Process case where order size <= book size"""
        filled_orders = []
        total_fill_price = 0
        current_quantity = 0
        split_level = None
        split_price = None
        
        for price_level in it:
            if current_quantity >= remaining_quantity:
                break
                
            level = book[price_level]
            level_quantity = level.quantity
            
            if current_quantity + level_quantity <= remaining_quantity:
                # Full level fill
                filled_orders.extend(list(level.orders))
                current_quantity += level_quantity
                total_fill_price += price_level * level_quantity
            else:
                # Partial level fill - need to split
                split_level = level
                split_price = price_level
                needed_quantity = remaining_quantity - current_quantity
                
                # Collect orders until we reach needed_quantity
                qty_from_level = 0
                orders_to_fill = []
                
                for order_idx in level.orders:
                    order_qty = self.orderArray.get_quantity(order_idx)
                    if qty_from_level + order_qty <= needed_quantity:
                        orders_to_fill.append(order_idx)
                        qty_from_level += order_qty
                    else:
                        # Split this order
                        remaining_for_order = needed_quantity - qty_from_level
                        self.orderArray.modify_quantity(order_idx, remaining_for_order)
                        orders_to_fill.append(order_idx)
                        break
                        
                filled_orders.extend(orders_to_fill)
                current_quantity += needed_quantity
                total_fill_price += price_level * needed_quantity
                break
        
        # Batch process all filled orders
        if filled_orders:
            self.fillLimitKeys(filled_orders)
        
        # Update book structure
        self._update_book_after_partial_fill(book, direction, filled_orders, split_level, split_price)
        
        return current_quantity, total_fill_price

    def _rebuild_book_after_fill(self, book, direction, price):
        """Efficiently rebuild book after large fill"""
        if direction == "buy":
            self.asks = SortedDict(
                (k, self.asks[k]) for k in self.asks.irange(minimum=price, maximum=None, inclusive=(False, False))
            )
        else:
            self.bids = SortedDict(
                (k, self.bids[k]) for k in self.bids.irange(minimum=price, maximum=None, inclusive=(False, False))
            ).__class__(lambda k: -k)

    def _update_book_after_partial_fill(self, book, direction, filled_orders, split_level, split_price):
        """Update book structure after partial fill"""
        # Remove fully filled price levels
        prices_to_remove = set()
        for order_idx in filled_orders:
            _, price, _ = self.orderArray.get_quantity_price_direction(order_idx)
            if price != split_price:  # Don't remove split level
                prices_to_remove.add(price)
        
        for price in prices_to_remove:
            del book[price]
        
        # Update size counters
        fill_quantity = sum(self.orderArray.get_quantity(idx) for idx in filled_orders)
        if direction == "buy":
            self._askSize -= fill_quantity
        else:
            self._bidSize -= fill_quantity

    def clearAgainstBook(self, quantity, direction, price, orderType, accountID):
        """
        ("`-''-/").___..--''"`-._ 
        `6_ 6  )   `-.  (     ).`-.__.`) 
        (_Y_.)'  ._   )  `._ `. ``-..-' 
        _..`--'_..-_/  /--'_.'
        ((((.-''  ((((.'  (((.-' 
        
        Congratulations for finding the secret sauce!
        This is my matching algorithm which tries to maximize batching,
        and is as efficient as I could make it... for now...
        - Nicolas Ollivier
        """
        # P.S. Ignore the inconsistent variable naming and 200+ lines of code in a single function

        remaining_quantity = quantity
        opp_direction = "sell" if direction == "buy" else "buy"
        dir_mult = 1 if direction == "buy" else -1
        total_fill_price = 0
        filled_keys = []  # List to track keys to remove after iteration

        # find quantity in opposite market book
        if direction == "buy": # buys -> sell book
            marketBook = self.marketSells
            marketQuantity = self.marketQuantity[1]
            book = self.asks
            bookSize = self.askSize
        else: # sells -> buy book
            marketBook = self.marketBuys
            marketQuantity = self.marketQuantity[0]
            book = self.bids
            bookSize = self.bidSize
        
        # predetermine fill price for market order operations based on order type
        if orderType == "market":
            fill_price = self.lastPrice
        else:
            fill_price = price

        """
        =^..^=   =^..^=   =^..^=    =^..^=    =^..^=    =^..^=    =^..^=
                    FILLING AGAINST MARKET ORDER BOOKS
        =^..^=   =^..^=   =^..^=    =^..^=    =^..^=    =^..^=    =^..^=
        """
        if marketQuantity > 0:
            if len(marketBook) == 0:  # Add this check
                marketQuantity = 0
            elif marketQuantity < remaining_quantity:
                # Convert marketBook deque to numpy array for vectorized operations
                market_book_array = np.array(list(marketBook))
                self.fillMarketKeys(market_book_array, fill_price)
                marketBook.clear() # all orders fill, so empty marketbook

                remaining_quantity -= marketQuantity
                total_fill_price += fill_price*marketQuantity
                marketQuantity = 0 # since market book is now empty

            else: # market book quantity >= quantity, so order will be fully filled by market orders
                market_orders = np.array(list(marketBook))
                cumulative_qty = 0
                split_idx = 0

                # get all quantities with one operation
                order_data = [(idx, *self.orderArray.get_account_quantity(idx)) 
                  for idx in market_orders]

                valid_orders = [(idx, acc, qty) for idx, acc, qty in order_data if qty > 0]

                if not valid_orders:
                    return 0, total_fill_price
                
                # Find split point for full vs partial fill
                for i, (idx, _, qty) in enumerate(valid_orders):
                    if cumulative_qty + qty >= remaining_quantity:
                        split_idx = i
                        break
                    cumulative_qty += qty
                
                # handle fully filled orders
                full_fill_orders = valid_orders[:split_idx]
                # Batch process fully filled orders
                full_fill_keys = [idx for idx, _, _ in full_fill_orders]
                self.fillMarketKeys(full_fill_keys, fill_price)
                
                # Update remaining quantity
                remaining_quantity -= cumulative_qty
                total_fill_price += fill_price * cumulative_qty
                
                # Handle the split order if it exists
                if split_idx < len(valid_orders):
                    split_idx, split_acc, split_qty = valid_orders[split_idx]
                    fill_qty = remaining_quantity
                    
                    # Modify the split order quantity
                    new_qty = self.orderArray.modify_quantity(split_idx, fill_qty)
                    
                    # Queue updates for split order
                    self.emissionQueue.append((split_acc, opp_direction, fill_qty))
                    self.accountUpdateQueue.append((split_acc, -dir_mult*fill_qty))
                    self.accountCashUpdateQueue.append((split_acc, dir_mult*fill_qty*fill_price))
                    
                    # Update market book
                    if new_qty > 0:
                        marketBook.clear()
                        marketBook.append(split_idx)
                        marketQuantity = new_qty
                    else:
                        marketBook.clear()
                        marketQuantity = 0
                    
                    total_fill_price += fill_price * fill_qty
                    remaining_quantity = 0
                
                # Process all queued updates
                self.process_queued_updates()
                
                return quantity, total_fill_price

        """
        =^..^=   =^..^=   =^..^=    =^..^=    =^..^=    =^..^=    =^..^=
                    FILLING AGAINST LIMIT ORDER BOOKS
        =^..^=   =^..^=   =^..^=    =^..^=    =^..^=    =^..^=    =^..^=
        """
        if bookSize > 0:
            # Get iterator for price levels
            it = book.irange(maximum=price) if orderType == "limit" else book
            
            # Process limit orders
            filled_quantity, fill_price = self.process_limit_orders(
                book, it, remaining_quantity, direction, orderType, fill_price
            )
            
            remaining_quantity -= filled_quantity
            total_fill_price += fill_price

        # Process all queued updates before returning
        self.process_queued_updates()
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
        if orderType == "market":
            # we want price 0 to be reserved for market orders to make other operations simpler
            price = 0
        else:
            if price == 0:
                price = 0.1 
            else:
                price = round(price, 2)

        # Fill as much as possible against other side of the book
        qFilled, totalFillPrice = self.clearAgainstBook(quantity, direction, price, orderType, accountID)
        
        translatedDirection = 1 if direction == "buy" else -1

        # Settle transacted quantity at total fill price
        emit_trade_update(accountID, self.asset, direction, qFilled)
        accounts[accountID].addPosition(self.asset, qFilled*translatedDirection)
        accounts[accountID].addPosition("CASH", - totalFillPrice*translatedDirection)

        # Handle the case where the order is completely filled
        if qFilled == quantity:
            return 0
        
        # Otherwise, remaining quantity, so add an order to the order book
        # Create order object
        order_qty = quantity - qFilled

        # Insert into array of orders 
        location = self.orderArray.insert(
            direction=direction,
            price=price,
            quantity=order_qty,
            accountID=accountID
        )

        # Adjust quantity stores accordingly and add to appropriate book
        if orderType == "market":
            if direction == "buy":
                self._marketBuyQuantity += order_qty
                self.marketBuys.append(location)
            else:
                self._marketSellQuantity += order_qty
                self.marketSells.append(location)
        else:
            if direction == "buy":
                self._bidSize += order_qty
                
                # if no OL exists, make one
                if price not in self.bids:
                    self.bids[price] = self.OrderLevel(location, order_qty)
                else:
                    self.bids[price].pushOrder(location, order_qty)

            else:
                self._askSize += order_qty

                # if no OL exists, make one
                if price not in self.asks:
                    self.asks[price] = self.OrderLevel(location, order_qty)
                else:
                    self.asks[price].pushOrder(location, order_qty)

        return location
        
    def cancelOrder(self, idx):
        qty, price, direction = self.orderArray.get_quantity_price_direction(idx)
        self.orderArray.remove(idx)

        if price == 0:
            if direction == 1: # BUY MARKET BOOK
                self.marketBuys.remove(idx)
                self._marketBuyQuantity -= qty
            else: # SELL MARKET BOOK
                self.marketSells.remove(idx)
                self._marketSellQuantity -= qty
        else:
            if direction == 1: # BIDS
                self._bidSize -= qty
                self._best_bid = None # uninitialize cache accordingly
                self.bids[price].remove(idx, qty)
            else: # ASKS
                self._askSize -= qty
                self._best_ask = None # uninitialize cache accordingly
                self.asks[price].remove(idx, qty)

    def cancelAllOldOrders(self):
        self.bids.clear()
        self._bidSize = 0
        self._best_bid = None
        self.asks.clear()
        self._askSize = 0
        self._best_ask = None
        self.marketBuys.clear()
        self.marketSells.clear()
        self._marketBuyQuantity = 0
        self._marketSellQuantity = 0

    def cancelOrdersByAccount(self, accountID):
        indices = self.orderArray.get_orders_by_account(accountID) 

        for idx in indices:
            self.cancelOrder(idx)

    def display(self):
        print(f"{self.asset} Orderbook")
        print(f"Last Price: {self.lastPrice}")
        print(f"Bids, Asks: {self.getBidAskPairs()}")
        if self._marketBuyQuantity>0:
            print(f"market Buy Quantity: {self._marketBuyQuantity}")
        if self._marketSellQuantity> 0 :
            print(f"market Sell Quantity: {self._marketSellQuantity}")

    def displayPrice(self):
        print(f"{self.asset} price: {round(self.lastPrice, 2)}")

    # PROPERTIES
    @property
    def bidSize(self):
        return self._bidSize

    @property
    def askSize(self):
        return self._askSize

    @property
    def bestBid(self):
        """Cached property for best bid price"""
        if not self.bids:
            return None
        if self._best_bid is None:
            first_key = next(iter(self.bids))  # Get first key directly
            self._best_bid = first_key
        return self._best_bid

    @property
    def bestAsk(self):
        """Cached property for best ask price"""
        if not self.asks:
            return None
        if self._best_ask is None:
            first_key = next(iter(self.asks))  # Get first key directly
            self._best_ask = first_key
        return self._best_ask
    
    @property
    def lastPrice(self):
        """Cached property for last price"""
        if not last_prices[self.asset]:
            return None
        if self._last_price is None:
            self._last_price = last_prices[self.asset]
        return self._last_price
    
    @property
    def marketQuantity(self):
        """Property for market buy quantity"""
        return self._marketBuyQuantity, self._marketSellQuantity

    @property
    def marketOrders(self):
        """Return """
        return self.marketBuys, self.marketSells
    
    @property
    def getUnfilledmarketOrders(self):
        return max(0, self._marketBuyQuantity), max(0, self._marketSellQuantity)
    
    @property 
    def getBids(self):
        if not self.bids:
            return None
        return self.bids

    @property
    def getAsks(self):
        if not self.asks:
            return None
        return self.asks
