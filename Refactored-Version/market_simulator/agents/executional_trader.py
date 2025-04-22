import random
from market_simulator.agents.base_agent import MarketAgent
from market_simulator.utils.market_utils import markets, last_prices

# ABSTRACT CLASS FOR TRADERS THAT EXECUTE ORDERS IN NON-TRIVIAL AMOUNTS
class ExecutionalTrader(MarketAgent):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)
        self.intendedOrders = {}
        self.intendedPosition = {}
        self.conditionalOrders = {}
        self.nbboOrders = {}

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

    def targetPosition(self, orderBook, direction, price, quantity):
        self.intendedPosition[orderBook] = {"direction": direction, "price": price, "quantity": quantity}
    
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
                quantity = int(min(random.randint(1, 10000), 5000*min(1, max(0.01, price_diff)), order["quantity"]))
        
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
            # print("test")
            position = self.intendedPosition[orderBook]
            direction = 1 if position["direction"] == "buy" else -1
            gap = position["quantity"]*direction-self.getPosition(orderBook.asset)
            nbboOrder = self.nbboOrders.get(orderBook)
            if nbboOrder:
                distance = orderBook.lastPrice - nbboOrder.price
                nbboQuantity = orderBook.getRemainingQuantity(nbboOrder)
            else:
                distance = 1
                nbboQuantity = -1

            if gap > 0: # need to strategically refresh buy orders
                q = min(random.randint(1,10000),gap)
                if distance > 0.1 or nbboQuantity <= 0: # if far from current bid, refresh order
                    if distance > 0.1 and nbboQuantity > 0:
                        orderBook.cancelOrder(nbboOrder)
                        # print("cancelling buy")
                    
                    order = self.placeOrder(orderBook, "buy", orderBook.get_best_bid() + 0.1, q, "limit")
                    self.nbboOrders[orderBook] = order
                    # print("PLACING BUY @", orderBook.bestBid+0.1, " with q:", q)
            
            elif gap < 0: # need to strategically refresh sell orders
                q = min(random.randint(1,10000),-gap)
                if distance < - 0.1 or nbboQuantity <= 0: # if far from current ask, refresh order
                    if distance < -0.1 and nbboQuantity > 0:
                        orderBook.cancelOrder(nbboOrder)
                    order = self.placeOrder(orderBook, "sell", orderBook.get_best_ask() - 0.1, q, "limit")
                    self.nbboOrders[orderBook] = order
                    # print("PLACING SELL @", orderBook.bestAsk-0.1, " with q:", q)


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