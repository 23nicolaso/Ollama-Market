from enum import IntEnum
from time import time

from sortedcontainers import SortedList

class Side(IntEnum):
    BUY = 1
    SELL = -1

class Order:
    __slots__ = ('id', 'acc_id', 'timestamp', 'remaining', 'side')
    def __init__(self, id: int, acc_id: int, quantity: int, side: Side):
        self.id: int = id
        self.acc_id: int = acc_id
        self.timestamp: time = int(time())
        self.remaining: int = quantity
        self.side: Side = side
    
    def __getType__(self):
        return self.__class__

class MarketOrder(Order):
    def __init__(self, id: int, acc_id: int, quantity: int, side: Side):
        super().__init__(id, acc_id, quantity, side)

    def __repr__(self):
        return f"Market Order: id:{self.id}, q:{self.remaining}, time:{self.timestamp}"

class LimitOrder(Order):
    __slots__ = ('price',)
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
        return f"Limit Order: id:{self.id}, q:{self.remaining}, p:{self.price}, time:{self.timestamp}, side:{self.side}\n"
    
    def __eq__(self, other):
        return self.id==other.id and self.price == other.price and self.side == other.side and self.timestamp == other.timestamp

ol = SortedList()

order_1 = LimitOrder(0, 0, 20, 99.9, -1)
order_2 = LimitOrder(1, 0, 20, 100.0, -1)
order_3 = LimitOrder(2, 0, 20, 100.1, -1)

ol.add(order_1)
ol.add(order_2)
ol.add(order_3)

print(ol)
ol.discard(order_3)
# ol.discard(order_2)
ol.discard(order_1)
print(ol)