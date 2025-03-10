'''
This is a highly optimized version of the order book that:
1. Eliminates Order class instances to reduce object creation/destruction overhead
2. Uses primitive data structures (arrays/lists) for order storage
3. Maintains the hybrid approach for excellent asymptotic complexity
4. Reduces memory allocations and garbage collection pressure
5. Maintains compatibility with original function names

This approach is particularly effective for high-frequency scenarios with
many small orders and frequent modifications.
'''

from market_simulator.utils.market_utils import last_prices, accounts
from market_simulator.price_server import emit_trade_update
from market_simulator.portfolio_utils import calculate_portfolio_status, emit_portfolio_update
from sortedcontainers import SortedDict
from collections import deque
from time import time
try:
    import numpy as np
    USE_NUMPY = True
except ImportError:
    USE_NUMPY = False

# Simple order ID counter
_next_order_id = 0
def next_order_id():
    global _next_order_id
    _next_order_id += 1
    return _next_order_id

# Order property indices for array-based storage
ORDER_ID = 0
PRICE = 1
QUANTITY = 2
ACCOUNT_ID = 3
DIRECTION = 4
TIMESTAMP = 5
ORDER_FIELDS = 6  # Total number of fields

class OrderQueue:
    '''Optimized replacement for OrderLevel with compatible interface'''
    __slots__ = ('price', 'total_quantity', 'orders', 'is_numpy', 'asset', 'direction')
    
    def __init__(self, price, quantity, asset, direction):
        self.price = price
        self.total_quantity = 0
        self.asset = asset
        self.direction = direction
        self.is_numpy = USE_NUMPY
        
        # Initialize orders storage - either numpy array (faster) or list of lists
        if self.is_numpy:
            # Start with a small pre-allocated array that will grow as needed
            self.orders = np.zeros((10, ORDER_FIELDS), dtype=object)
            self.orders[:, ORDER_ID] = -1  # Mark all as unused
        else:
            # Regular Python list-based storage
            self.orders = deque()
    
    # Original OrderLevel interface methods
    def isEmpty(self):
        """Check if the queue is empty - compatible with original"""
        return self.is_empty()
    
    def getPrice(self):
        """Get price - compatible with original"""
        return self.price
    
    def getQuantity(self):
        """Get total quantity - compatible with original"""
        return self.total_quantity
    
    def fulfillAll(self):
        """Fulfill all orders in this level - compatible with original"""
        self.fill_quantity(self.total_quantity)
        self.total_quantity = 0
    
    def cancelOldOrders(self):
        """Cancel old orders - compatible with original"""
        # In our implementation we don't track age, so just do cleanup
        self.cleanup()
    
    def cancelAll(self):
        """Cancel all orders - compatible with original"""
        if self.is_numpy:
            self.orders[:, ORDER_ID] = -1
        else:
            self.orders.clear()
        self.total_quantity = 0
    
    def fulfillQuantity(self, quantityToFill):
        """Fill quantity - compatible with original interface"""
        filled, _, _ = self.fill_quantity(quantityToFill)
        return 0 if filled >= quantityToFill else quantityToFill - filled
    
    def cancelOrdersFromID(self, accountID):
        """Cancel orders by account ID - compatible with original"""
        return self.cancel_by_account(accountID)
    
    def getAccountID(self):
        """Get account ID of first order - compatible with original"""
        if self.is_numpy:
            valid_indices = np.where(self.orders[:, ORDER_ID] != -1)[0]
            if len(valid_indices) > 0:
                return self.orders[valid_indices[0], ACCOUNT_ID]
        elif self.orders:
            return self.orders[0][ACCOUNT_ID]
        return None
    
    # New optimized implementation methods
    def add_order(self, direction, quantity, accountID):
        """Add an order - compatible with original interface"""
        dir_value = 1 if direction == 1 else -1
        order_id = next_order_id()
        timestamp = time()
        
        if self.is_numpy:
            # Find first unused slot or resize if needed
            unused_slots = np.where(self.orders[:, ORDER_ID] == -1)[0]
            
            if len(unused_slots) == 0:
                # Double array size when full
                old_size = self.orders.shape[0]
                new_orders = np.zeros((old_size * 2, ORDER_FIELDS), dtype=object)
                new_orders[:old_size] = self.orders
                new_orders[old_size:, ORDER_ID] = -1  # Mark new slots as unused
                self.orders = new_orders
                idx = old_size
            else:
                idx = unused_slots[0]
            
            # Store order data directly in array
            self.orders[idx, ORDER_ID] = order_id
            self.orders[idx, PRICE] = self.price
            self.orders[idx, QUANTITY] = quantity
            self.orders[idx, ACCOUNT_ID] = accountID
            self.orders[idx, DIRECTION] = dir_value
            self.orders[idx, TIMESTAMP] = timestamp
        else:
            # Add as a simple list
            order_data = [order_id, self.price, quantity, accountID, dir_value, timestamp]
            self.orders.append(order_data)
        
        self.total_quantity += quantity
        return order_id
    
    def fill_quantity(self, quantity_to_fill):
        """Fill orders from the queue and return executed trades"""
        if quantity_to_fill <= 0:
            return 0, [], set()
            
        filled_quantity = 0
        executed_trades = []
        accounts_to_update = set()
        
        if self.is_numpy:
            # Process valid orders (those with quantity > 0)
            valid_indices = np.where(
                (self.orders[:, ORDER_ID] != -1) & 
                (self.orders[:, QUANTITY] > 0)
            )[0]
            
            # Sort by timestamp if needed (usually already in order)
            if len(valid_indices) > 1:
                valid_indices = valid_indices[np.argsort(self.orders[valid_indices, TIMESTAMP])]
            
            remaining = quantity_to_fill
            removed_indices = []
            
            for idx in valid_indices:
                if remaining <= 0:
                    break
                
                # Get order data
                order_qty = self.orders[idx, QUANTITY]
                
                # Calculate fill amount
                fill_amount = min(remaining, order_qty)
                
                # Record trade
                executed_trades.append((
                    self.orders[idx, ACCOUNT_ID],
                    self.price, # Use level price
                    fill_amount,
                    self.orders[idx, DIRECTION]
                ))
                
                # Update account directly
                account_id = self.orders[idx, ACCOUNT_ID]
                accounts[account_id].tradeAtPrice(self.asset, self.price, fill_amount, self.direction)
                
                # Track account for later portfolio updates
                accounts_to_update.add(account_id)
                
                # Update counts
                filled_quantity += fill_amount
                remaining -= fill_amount
                self.orders[idx, QUANTITY] -= fill_amount
                self.total_quantity -= fill_amount
                
                # Mark completely filled orders for removal
                if self.orders[idx, QUANTITY] <= 0:
                    removed_indices.append(idx)
                
                # Update last price
                last_prices[self.asset] = self.price
            
            # Clear filled orders
            for idx in removed_indices:
                self.orders[idx, ORDER_ID] = -1  # Mark as unused
        else:
            # Use deque for non-numpy implementation
            while self.orders and quantity_to_fill > 0:
                order = self.orders[0]
                
                # Calculate fill amount
                fill_amount = min(quantity_to_fill, order[QUANTITY])
                
                # Update account directly
                account_id = order[ACCOUNT_ID]
                accounts[account_id].tradeAtPrice(self.asset, self.price, fill_amount, self.direction)
                
                # Record trade
                executed_trades.append((
                    account_id,
                    self.price,
                    fill_amount,
                    order[DIRECTION]
                ))
                
                # Track account for later portfolio updates
                accounts_to_update.add(account_id)
                
                # Update counts
                filled_quantity += fill_amount
                quantity_to_fill -= fill_amount
                order[QUANTITY] -= fill_amount
                self.total_quantity -= fill_amount
                
                # Remove completely filled orders
                if order[QUANTITY] <= 0:
                    self.orders.popleft()
                    
                # Update last price
                last_prices[self.asset] = self.price
            
        return filled_quantity, executed_trades, accounts_to_update
    
    def is_empty(self):
        """Check if the queue is empty"""
        if self.is_numpy:
            valid_indices = np.where(self.orders[:, ORDER_ID] != -1)[0]
            return len(valid_indices) == 0
        else:
            return len(self.orders) == 0
    
    def cancel_by_account(self, account_id):
        """Cancel all orders for a specific account"""
        cancelled_quantity = 0
        
        if self.is_numpy:
            # Find orders from this account
            account_indices = np.where(
                (self.orders[:, ORDER_ID] != -1) & 
                (self.orders[:, ACCOUNT_ID] == account_id)
            )[0]
            
            # Sum up quantities
            cancelled_quantity = np.sum(self.orders[account_indices, QUANTITY])
            
            # Mark as cancelled
            self.orders[account_indices, ORDER_ID] = -1
        else:
            new_orders = deque()
            
            for order in self.orders:
                if order[ACCOUNT_ID] == account_id:
                    cancelled_quantity += order[QUANTITY]
                else:
                    new_orders.append(order)
            
            self.orders = new_orders
        
        self.total_quantity -= cancelled_quantity
        return cancelled_quantity
    
    def cleanup(self):
        """Compact storage by removing cancelled orders"""
        if self.is_numpy and len(self.orders) > 100:  # Only compact large arrays
            valid_indices = np.where(self.orders[:, ORDER_ID] != -1)[0]
            if len(valid_indices) < self.orders.shape[0] // 2:
                # Compact if more than half the array is unused
                self.orders = self.orders[valid_indices]
                # Add some empty slots for future orders
                new_size = max(len(valid_indices) * 2, 10)
                new_orders = np.zeros((new_size, ORDER_FIELDS), dtype=object)
                new_orders[:len(valid_indices)] = self.orders
                new_orders[len(valid_indices):, ORDER_ID] = -1
                self.orders = new_orders

    # Compatibility with netQuantity property used in display()
    @property
    def netQuantity(self):
        return self.total_quantity


class OrderBook:
    def __init__(self, asset, initialPrice):
        self.asset = asset
        
        # Sorted dictionaries of price -> OrderQueue
        # For bids, we want highest price first
        self.bids = SortedDict(lambda price: -price)
        # For asks, we want lowest price first
        self.asks = SortedDict()
        
        # Direct storage for market orders to avoid objects
        # Format: (quantity, account_id, timestamp)
        self.urgentBuys = deque()  # Changed from market_buys to match original
        self.urgentSells = deque()  # Changed from market_sells to match original
        
        # Initialize price
        last_prices[asset] = initialPrice
        
        # Performance tracking
        self.trade_batch_size = 50  # Process in batches for efficiency
        self.pending_trades = []
        self.pending_updates = set()
        self.last_cleanup = time()
    
    # Original interface methods
    def getBids(self):
        """Get bids - compatible with original"""
        return self.bids
    
    def getAsks(self):
        """Get asks - compatible with original"""
        return self.asks
    
    def clearFarOrders(self):
        # delete this
        return
    
    def getMidPrice(self):
        """Get mid price - compatible with original"""
        return self.get_mid_price()
    
    def getUrgentOrders(self):
        """Get urgent orders - compatible with original"""
        buys = [(qty, acc) for qty, acc, _ in self.urgentBuys]
        sells = [(qty, acc) for qty, acc, _ in self.urgentSells]
        return buys, sells
    
    def cancelAllOldOrders(self):
        """Cancel all old orders - compatible with original"""
        self.cancel_all_orders()
    
    def cancelOrdersByAccount(self, accountID):
        """Cancel orders by account ID - compatible with original"""
        self.cancel_order_by_account(accountID)
    
    def clearEmptyOrderlevels(self):
        """Clear empty order levels - compatible with original"""
        self._cleanup_order_queues()
    
    def getUrgentQuantity(self):
        """Get total quantities of urgent orders - compatibility method"""
        buy_size, sell_size = self.get_market_order_size()
        return buy_size, sell_size
    
    def displayPrice(self):
        """Display last price - compatible with original"""
        print(f"{self.asset} price: {round(self.getLastPrice(), 2)}")
    
    def getBidSize(self):
        """Get bid size - compatible with original"""
        return self.get_bid_size()
    
    def getAskSize(self):
        """Get ask size - compatible with original"""
        return self.get_ask_size()
    
    def getBestBid(self):
        """Get best bid - compatible with original"""
        if not self.bids:
            return None
        price = max(self.bids.keys())
        return self.bids[price]
    
    def getBestAsk(self):
        """Get best ask - compatible with original"""
        if not self.asks:
            return None
        price = min(self.asks.keys())
        return self.asks[price]
    
    def matchBooks(self):
        """Match books - compatible with original"""
        # This functionality is handled differently in our implementation
        # It happens automatically when adding orders
        if self.bids and self.asks:
            return max(self.bids.keys()), min(self.asks.keys())
        return None, None
    
    def getUnfilledUrgentOrders(self):
        """Get unfilled urgent orders - compatible with original"""
        # Return empty lists if no urgent orders exist
        buys = [(qty, acc) for qty, acc, _ in self.urgentBuys]
        sells = [(qty, acc) for qty, acc, _ in self.urgentSells]
        return buys, sells
    
    def fillUrgentOrders(self):
        """Fill urgent orders - compatible with original"""
        # This delegates to our optimized market order matching method
        self._match_market_orders()
        return len(self.pending_trades) // 2  # Estimate filled orders
    
    def getLastPrice(self):
        """Get last price - compatible with original"""
        return last_prices.get(self.asset, None)
    
    # Implementation methods
    def _process_pending_updates(self):
        """Process any pending trade and portfolio updates"""
        if self.pending_trades:
            # Process trades
            for update in self.pending_trades:
                emit_trade_update(*update)
            self.pending_trades = []
            
            # Process portfolio updates
            for account_id in self.pending_updates:
                if account_id != "MARKET MAKER":  # Skip market maker for performance
                    portfolio_data = calculate_portfolio_status(accounts[account_id])
                    emit_portfolio_update(account_id, portfolio_data)
            self.pending_updates = set()
    
    def _emit_trade_events(self, executed_trades, contra_account_id, contra_direction):
        """Queue trade events for batch processing"""
        contra_dir_name = "buy" if contra_direction > 0 else "sell"
        
        for account_id, price, quantity, direction in executed_trades:
            dir_name = "buy" if direction > 0 else "sell"
            
            # Queue trade updates
            self.pending_trades.append((account_id, self.asset, dir_name, quantity, price))
            self.pending_trades.append((contra_account_id, self.asset, contra_dir_name, quantity, price))
            
            # Update account positions - already handled in OrderQueue.fill_quantity
            # accounts[account_id].addPosition(self.asset, quantity * direction)
            # accounts[account_id].addPosition("CASH", -price * quantity * direction)
            
            # Track accounts needing portfolio updates
            self.pending_updates.add(account_id)
            self.pending_updates.add(contra_account_id)
            
            # Update last price
            last_prices[self.asset] = price
            
        # Process updates if batch size reached
        if len(self.pending_trades) >= self.trade_batch_size:
            self._process_pending_updates()
    
    def add_limit_order(self, direction, price, quantity, account_id):
        """Add a limit order to the book"""
        # Try to match against the opposite book first
        opposite_book = self.asks if direction == "buy" else self.bids
        dir_value = 1 if direction == "buy" else -1
        executed_trades = []
        accounts_to_update = set()
        remaining_qty = quantity
        prices_to_remove = []
        
        # Check if order crosses the book
        for opp_price in list(opposite_book.keys()):
            # Break if price doesn't cross
            if (direction == "buy" and opp_price > price) or \
               (direction == "sell" and opp_price < price):
                break
                
            order_queue = opposite_book[opp_price]
            
            # Fill as much as possible from this price level
            filled_qty, trades, accounts = order_queue.fill_quantity(remaining_qty)
            
            if filled_qty > 0:
                executed_trades.extend(trades)
                accounts_to_update.update(accounts)
                remaining_qty -= filled_qty
                
                # Check if price level is empty
                if order_queue.is_empty() or order_queue.total_quantity == 0:
                    prices_to_remove.append(opp_price)
                
                # Break if order filled
                if remaining_qty <= 0:
                    break
        
        # Remove empty price levels
        for p in prices_to_remove:
            opposite_book.pop(p)
        
        # Process trades
        if executed_trades:
            self._emit_trade_events(executed_trades, account_id, dir_value)
        
        # Add remaining quantity to book
        if remaining_qty > 0:
            # Get the correct book
            book = self.bids if direction == "buy" else self.asks
            
            # Add to existing price level or create new one
            if price not in book:
                direction_value = 1 if direction == "buy" else -1
                book[price] = OrderQueue(price, remaining_qty, self.asset, direction_value)
            
            book[price].add_order(dir_value, remaining_qty, account_id)
        
        # Periodic cleanup
        current_time = time()
        if current_time - self.last_cleanup > 10:  # Every 10 seconds
            self._cleanup_order_queues()
            self.last_cleanup = current_time
            
        return quantity - remaining_qty  # Return filled quantity
    
    def add_market_order(self, direction, quantity, account_id):
        """Add a market order"""
        # Match immediately against the opposite book
        opposite_book = self.asks if direction == "buy" else self.bids
        dir_value = 1 if direction == "buy" else -1
        executed_trades = []
        accounts_to_update = set()
        remaining_qty = quantity
        prices_to_remove = []
        
        # For market orders, match against all available liquidity
        for opp_price in list(opposite_book.keys()):
            order_queue = opposite_book[opp_price]
            
            # Fill as much as possible
            filled_qty, trades, accounts = order_queue.fill_quantity(remaining_qty)
            
            if filled_qty > 0:
                executed_trades.extend(trades)
                accounts_to_update.update(accounts)
                remaining_qty -= filled_qty
                
                # Check if price level is empty
                if order_queue.is_empty() or order_queue.total_quantity == 0:
                    prices_to_remove.append(opp_price)
                
                # Break if order filled
                if remaining_qty <= 0:
                    break
        
        # Remove empty price levels
        for p in prices_to_remove:
            opposite_book.pop(p)
        
        # Process trades
        if executed_trades:
            self._emit_trade_events(executed_trades, account_id, dir_value)
        
        # Add any remaining quantity to market order queue
        if remaining_qty > 0:
            if direction == "buy":
                self.urgentBuys.append((remaining_qty, account_id, time()))
            else:
                self.urgentSells.append((remaining_qty, account_id, time()))
            
            # Try to match market orders with each other
            self._match_market_orders()
        
        return quantity - remaining_qty
    
    def _match_market_orders(self):
        """Match market buy and sell orders with each other"""
        # Process until no more matches possible
        while self.urgentBuys and self.urgentSells:
            buy_qty, buy_account, buy_time = self.urgentBuys[0]
            sell_qty, sell_account, sell_time = self.urgentSells[0]
            
            # Skip self-trades
            if buy_account == sell_account:
                if buy_qty > sell_qty:
                    self.urgentBuys[0] = (buy_qty - sell_qty, buy_account, buy_time)
                    self.urgentSells.popleft()
                elif sell_qty > buy_qty:
                    self.urgentSells[0] = (sell_qty - buy_qty, sell_account, sell_time)
                    self.urgentBuys.popleft()
                else:
                    self.urgentBuys.popleft()
                    self.urgentSells.popleft()
                continue
            
            # Calculate trade size and price
            trade_qty = min(buy_qty, sell_qty)
            
            # Determine price
            trade_price = self.get_mid_price() or last_prices[self.asset]
            
            # Execute trade
            accounts[buy_account].addPosition(self.asset, trade_qty)
            accounts[buy_account].addPosition("CASH", -trade_price * trade_qty)
            accounts[sell_account].addPosition(self.asset, -trade_qty)
            accounts[sell_account].addPosition("CASH", trade_price * trade_qty)
            
            # Queue trade updates
            self.pending_trades.append((buy_account, self.asset, "buy", trade_qty, trade_price))
            self.pending_trades.append((sell_account, self.asset, "sell", trade_qty, trade_price))
            
            # Track for portfolio updates
            self.pending_updates.add(buy_account)
            self.pending_updates.add(sell_account)
            
            # Update last price
            last_prices[self.asset] = trade_price
            
            # Update order quantities or remove filled orders
            if buy_qty > trade_qty:
                self.urgentBuys[0] = (buy_qty - trade_qty, buy_account, buy_time)
                self.urgentSells.popleft()
            elif sell_qty > trade_qty:
                self.urgentSells[0] = (sell_qty - trade_qty, sell_account, sell_time)
                self.urgentBuys.popleft()
            else:
                self.urgentBuys.popleft()
                self.urgentSells.popleft()
        
        # Process any pending updates
        if self.pending_trades:
            self._process_pending_updates()
    
    def addOrder(self, direction, price, quantity, orderType, accountID):
        """Add an order to the book - compatible with original"""
        # Validate inputs
        if orderType not in ["limit", "market"]:
            raise ValueError("Invalid order type: " + str(orderType))
        if direction not in ["buy", "sell"]:
            raise ValueError("Invalid direction: " + str(direction))
        if quantity <= 0:
            raise ValueError("Quantity must be positive: " + str(quantity))
        if accountID not in accounts:
            raise ValueError("Account ID not found: " + str(accountID))
        if price < 0 and orderType == "limit":
            print("Price must be positive: " + str(price))
            return
        
        # Process order based on type
        if orderType == "limit":
            return self.add_limit_order(direction, price, quantity, accountID)
        else:  # Market order
            return self.add_market_order(direction, quantity, accountID)
    
    def _cleanup_order_queues(self):
        """Periodic cleanup of order queues to optimize memory usage"""
        for queue in list(self.bids.values()):
            queue.cleanup()
            if queue.is_empty() or queue.total_quantity == 0:
                self.bids.pop(queue.price)
                
        for queue in list(self.asks.values()):
            queue.cleanup()
            if queue.is_empty() or queue.total_quantity == 0:
                self.asks.pop(queue.price)
    
    def cancel_order_by_account(self, account_id):
        """Cancel all orders for a specific account"""
        # Cancel limit orders
        for price, queue in list(self.bids.items()):
            cancelled = queue.cancel_by_account(account_id)
            if queue.is_empty() or queue.total_quantity == 0:
                self.bids.pop(price)
        
        for price, queue in list(self.asks.items()):
            cancelled = queue.cancel_by_account(account_id)
            if queue.is_empty() or queue.total_quantity == 0:
                self.asks.pop(price)
        
        # Cancel market orders
        self.urgentBuys = deque((qty, acc, ts) for qty, acc, ts in self.urgentBuys if acc != account_id)
        self.urgentSells = deque((qty, acc, ts) for qty, acc, ts in self.urgentSells if acc != account_id)
    
    def cancel_all_orders(self):
        """Cancel all orders in the book"""
        self.bids.clear()
        self.asks.clear()
        self.urgentBuys.clear()
        self.urgentSells.clear()
        self._process_pending_updates()  # Clear any pending updates
    
    # Utility methods
    def get_best_bid(self):
        """Get the highest bid price"""
        if not self.bids:
            return None
        return max(self.bids.keys())
    
    def get_best_ask(self):
        """Get the lowest ask price"""
        if not self.asks:
            return None
        return min(self.asks.keys())
    
    def get_mid_price(self):
        """Get the mid price"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return last_prices.get(self.asset)
    
    def get_bid_size(self):
        """Get total size of all bids"""
        return sum(queue.total_quantity for queue in self.bids.values())
    
    def get_ask_size(self):
        """Get total size of all asks"""
        return sum(queue.total_quantity for queue in self.asks.values())
    
    def get_market_order_size(self):
        """Get size of market orders"""
        buy_size = sum(qty for qty, _, _ in self.urgentBuys)
        sell_size = sum(qty for qty, _, _ in self.urgentSells)
        return buy_size, sell_size
    
    def display(self):
        """Display order book information - compatible with original"""
        print(f"{self.asset} Orderbook")
        print(f"Total Buy Orders: {len(self.bids)}")
        print(f"Total Sell Orders: {len(self.asks)}")
        print(f"Last Price: {self.getLastPrice()}")
        
        # Display bids
        bid_display = []
        for price in sorted(self.bids.keys(), reverse=True):
            bid_display.append(f"{self.bids[price].total_quantity} at {price}")
        print(f"Bids: {bid_display}")
        
        # Display asks
        ask_display = []
        for price in sorted(self.asks.keys()):
            ask_display.append(f"{self.asks[price].total_quantity} at {price}")
        print(f"Asks: {ask_display}")
        
        # Market orders
        buy_size, sell_size = self.get_market_order_size()
        print(f"Market Buy Orders: {buy_size}")
        print(f"Market Sell Orders: {sell_size}")
        
        print(f"Bid Size: {self.getBidSize()}")
        print(f"Ask Size: {self.getAskSize()}")