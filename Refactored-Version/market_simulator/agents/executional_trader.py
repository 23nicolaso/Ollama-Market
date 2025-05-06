import random
from market_simulator.agents.base_agent import MarketAgent
from market_simulator.utils.market_utils import markets, last_prices, price_history
from market_simulator.config import MM_BASE_ORDER_SIZE

# ABSTRACT CLASS FOR TRADERS THAT EXECUTE ORDERS IN NON-TRIVIAL AMOUNTS
class ExecutionalTrader(MarketAgent):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)
        self.intendedOrders = {}
        self.intendedPosition = {}
        self.icebergOrders = {}
        self.conditionalOrders = {}
        self.nbboOrders = {}

    def sniperAlgo(self, orderbook):
        """
        Hyper aggressive liquidity taking algorithm - iceberg detection, exploits them for fast fill.
        Uses pinging, lvl 2 data. 
        """
        if orderbook in self.intendedPosition:
            position = self.intendedPosition[orderbook]
            direction = 1 if position["direction"] == "buy" else -1
            gap = position["quantity"]*direction-self.getPosition(orderbook.asset)

            if gap > 0:
                previous_best_ask = orderbook.get_best_ask_quantity()
                self.placeOrder(orderbook, "buy", 0, 1, "market") # ping
                new_best_ask = orderbook.get_best_ask_quantity()
                if previous_best_ask == new_best_ask: # PONG (aka ICEBERG DETECTED)
                    self.placeOrder(orderbook, "buy", new_best_ask, int(gap), "limit")
                    self.cancelAllOrders(orderbook) # FILL AS MUCH Q AS POSSIBLE
                else: # otherwise just run as an opportunistic algo
                    self.opportunisticAlgo(orderbook)

            elif gap < 0:
                previous_best_bid = orderbook.get_best_bid_quantity()
                self.placeOrder(orderbook, "sell", 0, 1, "market") # ping
                new_best_bid = orderbook.get_best_bid_quantity()
                if previous_best_bid == new_best_bid: # PONG (aka ICEBERG DETECTED)
                    self.placeOrder(orderbook, "sell", new_best_bid, int(-gap), "limit")
                    self.cancelAllOrders(orderbook) # FILL AS MUCH Q AS POSSIBLE
                else: # otherwise just run as an opportunistic algo
                    self.opportunisticAlgo(orderbook)

    def stealthAlgo(self, orderbook):
        """
        Splits order into small, randomized quantities, trades on randomized intervals, 
        Pretends to be market maker, but joins one side with iceberg, puts other side deep in the book.
        More aggressive when price is attractive, less aggressive when price is not attractive
        """
        if orderbook in self.intendedPosition:
            position = self.intendedPosition[orderbook]
            direction = 1 if position["direction"] == "buy" else -1
            gap = position["quantity"]*direction-self.getPosition(orderbook.asset)
            avg_price = price_history[orderbook.asset].mean(n = 500)
            self.cancelAllOrders(orderbook)
            if gap > 0 and orderbook.last_price < avg_price: # needs to buy
                # modify aggression based on price
                proportion = 0.03 + max(-0.02, min(0.02, (position["price"] - orderbook.last_price)*0.01))

                cover_qty = min(MM_BASE_ORDER_SIZE, gap)
                mx = max(MM_BASE_ORDER_SIZE, int(gap*proportion))
                real_qty = min(random.randint(MM_BASE_ORDER_SIZE, mx), gap)
                self.placeOrder(orderbook, "buy", orderbook.get_best_bid(), int(real_qty), "iceberg", int(cover_qty))
                self.placeOrder(orderbook, "sell", orderbook.get_best_ask()+0.05, int(cover_qty), "limit")

            elif gap < 0 and orderbook.last_price > avg_price: # needs to sell
                proportion = 0.03 + max(-0.02, min(0.02, (orderbook.last_price - position["price"])*0.01))

                cover_qty = min(MM_BASE_ORDER_SIZE, int(-gap))
                mx = max(MM_BASE_ORDER_SIZE, int(-gap*proportion))
                real_qty = min(random.randint(MM_BASE_ORDER_SIZE, mx), -gap)
                self.placeOrder(orderbook, "sell", orderbook.get_best_ask(), int(real_qty), "iceberg", int(cover_qty))
                self.placeOrder(orderbook, "buy", orderbook.get_best_bid()-0.05, int(cover_qty), "limit")
            

    def opportunisticAlgo(self, orderbook):
        """
        Uses average price, spread, depth, volume spikes. 
        Only trades when there is high liquidity at attractive prices.
        """
        if orderbook in self.intendedPosition:
            position = self.intendedPosition[orderbook]
            direction = 1 if position["direction"] == "buy" else -1
            gap = position["quantity"]*direction-self.getPosition(orderbook.asset)
            avg_price = price_history[orderbook.asset].mean(n=200)

            if gap > 0:
                top_ask_q = orderbook.get_best_ask_quantity()
                ask_size = orderbook.ask_size
                if orderbook.last_price < avg_price or orderbook.last_price < position["price"] - 1:
                    qty = int(min(gap * 0.05, top_ask_q * 5, ask_size))
                    if ask_size >= 30000:
                        self.placeOrder(orderbook, "buy", 1, qty, "market")

                    else:
                        self.placeOrder(orderbook, "buy", orderbook.get_best_ask(), qty, "iceberg", top_ask_q)

            elif gap < 0:
                top_bid_q = orderbook.get_best_bid_quantity()
                bid_size = orderbook.bid_size
                if orderbook.last_price > avg_price or orderbook.last_price > position["price"] + 1:
                    qty = int(min(-gap*0.01, top_bid_q * 5, bid_size))
                    if bid_size >= 30000:
                        self.placeOrder(orderbook, "sell", 1, qty, "market")
                    
                    else:
                        self.placeOrder(orderbook, "sell", orderbook.get_best_bid(), qty, "iceberg", top_bid_q)

    def getConditionalOrdersInDirection(self, orderBook, direction):
        if orderBook not in self.conditionalOrders:
            return 0
        orders = [order for order in self.conditionalOrders[orderBook] if order["direction"] == direction]
        return sum(order["quantity"] for order in orders)

    def placeConditionalOrder(self, orderBook, direction, price, quantity, priceCondition, conditionalDirection):
        if orderBook.asset not in self.conditionalOrders:
            self.conditionalOrders[orderBook.asset] = []
        self.conditionalOrders[orderBook.asset].append({"direction": direction, "price": price, "quantity": quantity, "priceCondition": priceCondition, "conditionalDirection": conditionalDirection})

    def checkConditionalOrders(self, ticker):
        if ticker not in self.conditionalOrders:
            return
        for order in self.conditionalOrders[ticker]:
            if (order["conditionalDirection"] == "above" and last_prices[ticker] >= order["priceCondition"]) or (order["conditionalDirection"] == "below" and last_prices[ticker] <= order["priceCondition"]):
                self.placeOrder(markets[ticker], order["direction"], order["price"], order["quantity"], "market")
                self.conditionalOrders[ticker].remove(order)

    def executeTradeInLegs(self, orderBook, direction, price, quantity):
        self.intendedOrders[orderBook] = {"direction": direction, "price": price, "quantity": quantity}

    def targetPosition(self, orderBook, direction, price, initial_price, quantity, market_mode = False):
        self.intendedPosition[orderBook] = {"direction": direction, "price": price, "initial_price": initial_price, "quantity": quantity, "market_mode": market_mode}
    
    def removeOldIntendedOrders(self):
        self.intendedOrders.clear()

    def flattenIntendedExecutions(self):
        self.removeOldIntendedOrders()

    def removePositionTargets(self):
        self.intendedPosition.clear()
    
    def updatePositioning(self, market):
        self.partialExecuteMarket(markets[market])

    def partialExecuteMarket(self, orderBook): # for iceberg orders
        if orderBook in self.intendedOrders:
            order = self.intendedOrders[orderBook]
            if order["quantity"] > 0:
                # Use price as a reference, split up quantity into random orders
                # scale quantity on distance from price
                target_price = order["price"]
                price_diff = target_price - orderBook.last_price if order["direction"] == "sell" else orderBook.last_price - target_price
                quantity = int(min(random.randint(1, 1000), 1000*min(1, max(0.01, price_diff)), order["quantity"]))
        
                # Place the order
                if order["direction"] == "sell":
                    orderType = random.choice(["market", "limit"])
                    self.placeOrder(orderBook, "sell", orderBook.get_best_ask() + random.choice([-0.2, -0.1, 0, 0.1, 0.2]), quantity, orderType)
                else:
                    orderType = random.choice(["market", "limit"])
                    self.placeOrder(orderBook, "buy", orderBook.get_best_bid() + random.choice([-0.2, -0.1, 0, 0.1, 0.2]), quantity, orderType)

                # Update the remaining quantity
                order["quantity"] -= quantity
                
                # If the order is completely filled, remove it from intended orders
                if order["quantity"] <= 0:
                    del self.intendedOrders[orderBook]

    def refreshNBBOOrder(self, orderBook):
        if orderBook in self.intendedPosition:
            position = self.intendedPosition[orderBook]
            direction = 1 if position["direction"] == "buy" else -1
            gap = position["quantity"]*direction-self.getPosition(orderBook.asset)
            diff = orderBook.last_price - position["price"]
            last_price = orderBook.last_price
            if gap > 0: # need to strategically refresh buy orders
                p = min(0.1, (position["price"]-orderBook.last_price)/2)
                if orderBook.ask_size > 1000:
                    q = int(min(10000, random.randint(1000,orderBook.ask_size),gap))
                else:
                    q = int(min(random.randint(500,10000),gap))
                if diff <= 0 or position["market_mode"]:
                    self.cancelAllOrders(orderBook)
                    if random.random() < 0.3:
                        self.placeOrder(orderBook, "sell", 100, 100, "market") # Throw in small orders to add noise
                    else:
                        self.placeOrder(orderBook, "buy", 100, 100, "market") # Throw in small orders to add noise
                    if random.random() < p:
                        if diff <= -1:
                            self.placeOrder(orderBook, "buy", orderBook.get_best_ask(), q, "limit")
                        else:
                            self.placeOrder(orderBook, "buy", orderBook.get_best_bid(), q, "limit")

            elif gap < 0: # need to strategically refresh sell orders
                p = min(0.1, (orderBook.last_price-position["price"])/2)
                if orderBook.bid_size > 1000:
                    q = int(min(10000, random.randint(1000,orderBook.bid_size),-gap))
                else:
                    q = int(min(random.randint(500,10000),-gap))
                if diff >= 0 or position["market_mode"]:
                    self.cancelAllOrders(orderBook)    
                    if random.random() < 0.3:
                        self.placeOrder(orderBook, "sell", 100, 100, "market") # Throw in small orders to add noise
                        self.placeOrder(orderBook, "buy", 100, 100, "market") # Throw in small orders to add noise
                    if random.random() < p:
                        if diff >= 1:
                            self.placeOrder(orderBook, "sell", orderBook.get_best_bid(), q, "limit")
                        else:
                            self.placeOrder(orderBook, "sell", orderBook.get_best_ask(), q, "limit")

    def strategic_iceberg_update(self, orderBook, simulation_age):
        if orderBook in self.intendedPosition:
            position = self.intendedPosition[orderBook]
            direction = 1 if position["direction"] == "buy" else -1
            gap = position["quantity"]*direction-self.getPosition(orderBook.asset)
            avg_volume = orderBook.net_volume / min(1,simulation_age)
            last_price = orderBook.last_price
            
            if gap > 0: # need to strategically buy 
                if last_price <= position["price"]: # Buy EXTREMELY HARD
                    q = int(min(avg_volume, gap, 100000))
                    self.placeOrder(orderBook, "buy", orderBook.get_best_ask()+0.05, q, "limit")
        
            elif gap < 0: # need to strategically sell 
                if last_price >= position["price"]: # sell EXTREMELY HARD
                    q = int(min(avg_volume, -gap, 100000))
                    self.placeOrder(orderBook, "sell", orderBook.get_best_bid()-0.05, q, "limit")


    def updateOrdersInLegs(self, orderBook):
        if orderBook.get_best_bid() is None or orderBook.get_best_ask() is None:
            return
        try:
            if self.intendedOrders[orderBook]["direction"] == "buy" and self.intendedOrders[orderBook]["price"] >= orderBook.get_best_ask():
                quantityToFill = min(self.intendedOrders[orderBook]["quantity"],max(orderBook.get_best_ask_quantity(), 1000))
                self.placeOrder(orderBook, "buy", orderBook.get_best_ask(), quantityToFill, "limit")
                if quantityToFill > 5000:
                    print("WHAT")
                self.intendedOrders[orderBook]["quantity"] -= quantityToFill
            elif self.intendedOrders[orderBook]["direction"] == "sell" and self.intendedOrders[orderBook]["price"] <= orderBook.get_best_bid():
                quantityToFill = min(self.intendedOrders[orderBook]["quantity"], max(orderBook.get_best_bid_quantity, 1000))
                self.placeOrder(orderBook, "sell", orderBook.get_best_bid(), quantityToFill, "limit")
                if quantityToFill > 5000:
                    print("WHAT")
                self.intendedOrders[orderBook]["quantity"] -= quantityToFill
        except:
            return 