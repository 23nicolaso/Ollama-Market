"""
sortedListOrderBook.pyx
 - A reworked orderbook using sortedlist. 
 
The code here is heavily inspired and modified from this matching engine:
https://github.com/ridulfo/order-matching-engine/blob/main/ordermatchinengine/Order.py
"""

from time import time
from collections import deque

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
    cdef public int id
    cdef public int acc_id
    cdef public int timestamp
    cdef public int remaining
    cdef public Side side

    __slots__ = ('id', 'acc_id', 'timestamp', 'remaining', 'side')
    def __init__(self, id: int, acc_id: int, quantity: int, side: Side):
        self.id: int = id
        self.acc_id: int = acc_id
        self.timestamp: time = int(time())
        self.remaining: int = quantity
        self.side: Side = side
    
    def __getType__(self):
        return self.__class__

cdef class MarketOrder(Order):
    def __init__(self, id: int, acc_id: int, quantity: int, side: Side):
        super().__init__(id, acc_id, quantity, side)

    def __repr__(self):
        return f"Market Order: id:{self.id}, q:{self.quantity}, time:{self.timestamp}"

cdef class LimitOrder(Order):
    cdef public double price 

    __slots__ = ('price')
    def __init__(self, id: int, acc_id: int, quantity: int, price: float, side: Side):
        super().__init__(id, acc_id, quantity, side)
        self.price = price

    def __lt__(self, other):
        # price-time priority
        if self.price != other.price:
            return self.price > other.price if self.side == Side.BUY \
                else self.price < other.price
        else:
            return self.timestamp < other.timestamp

    def __repr__(self):
        return f"Limit Order: id:{self.id}, q:{self.quantity}, \
            p:{self.price}, time:{self.timestamp}, side:{self.side}"

cdef class OrderBook:
    """
    Orderbook to process orders
    """
    cdef str asset
    cdef double last_price
    cdef int bid_size, ask_size
    cdef object bids, asks, market_buys, market_sells, update_queue

    def __init__(self, asset: str, initialPrice: float):
        self.last_price: float = initialPrice
        self.asset: str = asset
        self.bid_size: int = 0
        self.ask_size: int = 0

        self.bids: SortedList[Order] = SortedList()
        self.asks: SortedList[Order] = SortedList()
        self.market_buys: deque[Order] = deque()
        self.market_sells: deque[Order] = deque()
        self.update_queue: list[tuple] = deque()

    cpdef void settle_trade(self, float price, int quantity, int acc_id_1, int acc_id_2, object direction):
        """
        Transfer the assets between buyer and seller
        """
        
        self.last_price = price
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

    cpdef void clear_update_queue(self):
        """
        Clear the update queue by settling all trades
        """
        while self.update_queue:
            self.settle_trade(*self.update_queue.popleft())

    cpdef public bint process_order(self, Order order):
        """
        Add the order to the book, and maintains book
        as necessary. Returns success/fail code.
        """

        # reject negative quantity orders
        if order.remaining <= 0: 
            return False

        cdef Side side = <Side>order.side
        cdef object opposite_market_book = self.market_buys if side == Side.SELL \
            else self.market_sells
        cdef object opposite_limit_book = self.bids if side == Side.SELL \
            else self.asks

        cdef float fill_price = self.last_price if isinstance(order, MarketOrder) \
            else order.price # price when filling against the market book
        
        cdef Order opposite_order
        while len(opposite_market_book) > 0:
            opposite_order = opposite_market_book.popleft()
            if opposite_order.remaining >= order.remaining:
                self.update_queue.append((fill_price, order.remaining, order.acc_id, opposite_order.acc_id, side))
                opposite_order.remaining -= order.remaining
                if opposite_order.remaining > 0:
                    opposite_market_book.appendleft(opposite_order)
                self.clear_update_queue()
                return True
            else:
                self.update_queue.append((fill_price, opposite_order.remaining, order.acc_id, opposite_order.acc_id, side))
                order.remaining -= opposite_order.remaining

        cdef bint is_market_order = isinstance(order, MarketOrder)

        while (len(opposite_limit_book) > 0 and ((not is_market_order\
            and (side == Side.BUY and order.price >= opposite_limit_book[0].price) \
            or (side == Side.SELL and order.price <= opposite_limit_book[0].price))
            or is_market_order)):
            opposite_order = opposite_limit_book.pop(0)
            if opposite_order.remaining >= order.remaining:
                self.update_queue.append((fill_price, order.remaining, order.acc_id, opposite_order.acc_id, side))

                opposite_order.remaining -= order.remaining
                if opposite_order.remaining > 0:
                    opposite_market_book.appendleft(opposite_order)
                self.clear_update_queue()
                return True
            else:
                self.update_queue.append((fill_price, opposite_order.remaining, order.acc_id, opposite_order.acc_id, order.side))
                order.remaining -= opposite_order.remaining

        if order.remaining > 0: # Add to books if not filled already
            self.clear_update_queue()
            if side == Side.BUY:
                if order.__class__ == LimitOrder:
                    self.bids.add(order)
                else:
                    self.market_buys.append(order)
            else:
                if order.__class__ == LimitOrder:
                    self.asks.add(order)
                else:
                    self.market_sells.append(order)
            return True

    cpdef cancel_order(self, int order_id, object book_id):
        if book_id == Book.ASKS:
            for order in self.asks:
                if order.id == order_id:
                    self.asks.discard(order)
                    return True
            return False
        if book_id == Book.BIDS:
            for order in self.bids:
                if order.id == order_id:
                    self.bids.discard(order)
                    return True
            return False

        if book_id == Book.MARKET_SELLS:
            for order in self.market_sells:
                if order.id == order_id:
                    self.market_sells.remove(order)
                    return True
            return False

        if book_id == Book.MARKET_BUYS:
            for order in self.market_buys:
                if order.id == order_id:
                    self.market_buys.remove(order)
                    return True
            return False

    def __repr__(self):
        lines = []
        lines.append("-"*5 + "OrderBook" + "-"*5)

        lines.append("\n Asks:")
        asks = self.asks.copy()
        while len(asks) > 0:
            lines.append(str(asks.pop()))

        lines.append("\t"*3 + "Bids:")
        bids = list(reversed(self.bids.copy()))
        while len(bids) > 0:
            lines.append("\t"*3 + str(bids.pop()))

        lines.append("-"*20)
        return "\n".join(lines)