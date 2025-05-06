"""
sortedListOrderBook.pyx
 - A reworked orderbook using sortedlist. 
 
Although I made several tweaks,
The code here is greatly inspired by this matching engine:
https://github.com/ridulfo/order-matching-engine/
"""

from time import time
from collections import deque
import sys

from sortedcontainers import SortedList

from market_simulator.utils.market_utils import accounts

cdef enum Side:
    BUY = 1
    SELL = -1

cdef enum Book:
    BIDS = 0
    ASKS = 1
    MARKET_BUYS = 2
    MARKET_SELLS = 3

cdef class Order:
    cdef public int acc_id
    cdef public double timestamp
    cdef public int remaining
    cdef public Side side

    __slots__ = ('id', 'acc_id', 'timestamp', 'remaining', 'side')
    def __init__(self, acc_id: int, quantity: int, side: Side):
        self.acc_id: int = acc_id
        self.timestamp: double = time()
        self.remaining: int = quantity
        self.side: Side = side
    
    def __getType__(self):
        return self.__class__

cdef class MarketOrder(Order):
    def __init__(self, acc_id: int, quantity: int, side: Side):
        super().__init__(acc_id, quantity, side)

    def __repr__(self):
        return f"Market Order: acc_id:{self.acc_id}, q:{self.remaining}, time:{self.timestamp}"

cdef class LimitOrder(Order):
    cdef public double price 

    __slots__ = ('price',)
    def __init__(self, acc_id: int, quantity: int, price: float, side: Side):
        super().__init__(acc_id, quantity, side)
        self.price = price

    def __lt__(self, other):
        # price-time priority
        if self.price != other.price:
            return self.price > other.price if self.side == Side.BUY \
                else self.price < other.price
        else:
            return self.timestamp < other.timestamp

    def __repr__(self):
        return f"Limit Order: acc_id:{self.acc_id}, q:{self.remaining}, \
            p:{self.price}, time:{self.timestamp}, side:{self.side}"
    
    def __eq__(self, other):
        return self.price == other.price and self.side == other.side and self.timestamp == other.timestamp

cdef class IcebergOrder(LimitOrder):
    """Effectively just a limit order, but when displaying it, the quantity shown is visible_size instead"""
    cdef int max_visible_size
    cdef public int displayed_size

    __slots__ = ('max_visible_size','displayed_size', )
    def __init__(self, acc_id: int, remaining: int, price: float, side: Side, max_visible_size: int):
        super().__init__(acc_id, remaining, price, side)
        self.max_visible_size = max_visible_size
        self.displayed_size = min(self.max_visible_size, self.remaining) 

    def refresh(self):
        self.displayed_size = min(self.max_visible_size, self.remaining)
    
    def __repr__(self):
        return f"Iceberg Order: acc_id:{self.acc_id}, q:{self.remaining}, \
            p:{self.price}, time:{self.timestamp}, side:{self.side}, max_visible:{self.max_visible_size}"

cdef class OrderBook:
    """
    Orderbook to process orders
    """
    cdef public int net_volume
    cdef public str asset
    cdef public double last_price
    cdef public int bid_size, ask_size
    cdef object bids, asks, market_buys, market_sells, update_queue

    def __init__(self, asset: str, initialPrice: float):
        self.last_price: float = initialPrice
        self.asset: str = asset
        self.bid_size: int = 0
        self.ask_size: int = 0
        self.net_volume: int = 0

        self.bids: SortedList[Order] = SortedList()
        self.asks: SortedList[Order] = SortedList()
        self.market_buys: deque[Order] = deque()
        self.market_sells: deque[Order] = deque()
        self.update_queue: list[tuple] = deque()

    cpdef int get_best_bid_quantity(self):
        if self.bids:
            if self.bids[0].__class__ == IcebergOrder:
                self.bids[0].refresh()
                return self.bids[0].displayed_size
            else:
                return self.bids[0].remaining
        else:
            return 0
        
    cpdef int get_best_ask_quantity(self):
        if self.asks:
            if self.asks[0].__class__ == IcebergOrder:
                self.asks[0].refresh()
                return self.asks[0].displayed_size
            else:
                return self.asks[0].remaining
        else: 
            return 0

    cpdef int get_market_buy_quantity(self):
        cdef int return_value = 0
        for order in self.market_buys:
            return_value += order.remaining
        
        return return_value
    
    cpdef int get_market_sell_quantity(self):
        cdef int return_value = 0
        for order in self.market_sells:
            return_value += order.remaining

        return return_value

    cpdef double get_best_bid(self):
        """
        Get price of the best bid
        """
        if self.bids:
            return self.bids[0].price
        else:
            return self.last_price
    
    cpdef double get_best_ask(self):
        """
        Get price of best ask
        """
        if self.asks:
            return self.asks[0].price
        else:
            return self.last_price

    cpdef object get_bids(self):   
        cdef object bids_dict = {}
        cdef object order
        cdef int qty
        for order in self.bids:
            if order.__class__ == IcebergOrder:
                order.refresh()
                qty = order.displayed_size
            else:
                qty = order.remaining

            if bids_dict.get(str(order.price)):
                bids_dict[str(order.price)] += qty
            else:
                bids_dict[str(order.price)] = qty

        return bids_dict

    cpdef object get_asks(self):    
        cdef object asks_dict = {}
        cdef object order
        cdef int qty
        for order in self.asks:
            if order.__class__ == IcebergOrder:
                order.refresh()
                qty = order.displayed_size
            else:
                qty = order.remaining
                
            if asks_dict.get(str(order.price)):
                asks_dict[str(order.price)] += qty
            else:
                asks_dict[str(order.price)] = qty

        return asks_dict

    cpdef void settle_trade(self, float price, int quantity, int acc_id_1, int acc_id_2, object direction):
        """
        Transfer the assets between buyer and seller
        """
        self.net_volume += quantity
        
        if direction ==  Side.BUY:
            accounts[acc_id_1].tradeAtPrice(self.asset, price,
                quantity, 1)
                
            accounts[acc_id_2].tradeAtPrice(self.asset, price,
                quantity, -1)
        elif direction == Side.SELL:
            accounts[acc_id_1].tradeAtPrice(self.asset, price,
                quantity, -1)
                
            accounts[acc_id_2].tradeAtPrice(self.asset, price,
                quantity, 1)

    cpdef public void process_order(self, Order order):
        """
        Add the order to the book, and maintains book
        as necessary. Returns success/fail code.
        """

        # reject negative quantity orders
        if order.remaining <= 0: 
            return
        
        cdef Side side = <Side>order.side
        cdef object opposite_market_book = self.market_buys if side == Side.SELL \
            else self.market_sells
        cdef object opposite_limit_book = self.bids if side == Side.SELL \
            else self.asks

        cdef float fill_price = self.last_price if order.__class__ == MarketOrder \
            else order.price # price when filling against the market book
        
        cdef Order opposite_order
        while len(opposite_market_book) > 0:
            opposite_order = opposite_market_book.popleft()
            if opposite_order.remaining >= order.remaining:
                self.last_price = fill_price
                self.settle_trade(fill_price, order.remaining, order.acc_id, opposite_order.acc_id, side)
                # self.update_queue.append((fill_price, order.remaining, order.acc_id, opposite_order.acc_id, side))
                opposite_order.remaining -= order.remaining
                if opposite_order.remaining > 0:
                    opposite_market_book.appendleft(opposite_order)
                return
            else:
                self.last_price = fill_price
                self.settle_trade(fill_price, opposite_order.remaining, order.acc_id, opposite_order.acc_id, side)
                # self.update_queue.append((fill_price, opposite_order.remaining, order.acc_id, opposite_order.acc_id, side))
                order.remaining -= opposite_order.remaining

        cdef bint is_market_order = order.__class__ == MarketOrder

        while (len(opposite_limit_book) > 0 
                and (is_market_order 
                or (not is_market_order\
                    and (side == Side.BUY and order.price >= opposite_limit_book[0].price) \
                    or (side == Side.SELL and order.price <= opposite_limit_book[0].price)))):
            opposite_order = opposite_limit_book.pop(0)
            if opposite_order.remaining >= order.remaining:
                self.last_price = opposite_order.price
                self.settle_trade(opposite_order.price, order.remaining, order.acc_id, opposite_order.acc_id, side)
                # self.update_queue.append((opposite_order.price, order.remaining, order.acc_id, opposite_order.acc_id, side))

                opposite_order.remaining -= order.remaining
                if opposite_order.remaining > 0:
                    opposite_limit_book.add(opposite_order)
                return 
            else:
                self.last_price = opposite_order.price
                self.settle_trade(opposite_order.price, opposite_order.remaining, order.acc_id, opposite_order.acc_id, side)
                # self.update_queue.append((opposite_order.price, opposite_order.remaining, order.acc_id, opposite_order.acc_id, order.side))
                order.remaining -= opposite_order.remaining

        if order.remaining > 0: # Add to books if not filled already
            if side == Side.BUY:
                if order.__class__ == LimitOrder or order.__class__ == IcebergOrder:
                    self.bids.add(order)
                else:
                    self.market_buys.append(order)
            else:
                if order.__class__ == LimitOrder or order.__class__ == IcebergOrder:
                    self.asks.add(order)
                else:
                    self.market_sells.append(order)
            
            return 

    cpdef cancel_orders_from_account(self, int acc_id):
        """
        Simple implementation to cancel all orders from an account
        Slightly inefficient O(n + k*log(n)) 
        Much better to cancel orders individually from account 
        for O(k*log(n)) performance.
        """
        for order in list(self.asks): 
            if order.acc_id == acc_id:
                self.asks.discard(order)
            
        for order in list(self.bids):
            if order.acc_id == acc_id:
                self.bids.discard(order)

    cpdef cancel_order(self, Order order):
        self.asks.discard(order)
        self.bids.discard(order)

    def __repr__(self):
        lines = []
        lines.append("-"*5 + "OrderBook" + "-"*5)

        lines.append("\n Asks:")
        asks = self.asks.copy()
        while len(asks) > 0:
            lines.append(str(asks.pop()))

        lines.append("\n"*3 + "Bids:")
        bids = list(reversed(self.bids.copy()))
        while len(bids) > 0:
            lines.append(str(bids.pop()))

        lines.append("-"*20)
        return "\n".join(lines)