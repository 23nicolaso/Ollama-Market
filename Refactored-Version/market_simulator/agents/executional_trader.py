import random
from market_simulator.agents.base_agent import MarketAgent
from market_simulator.utils.market_utils import markets, last_prices

# ABSTRACT CLASS FOR TRADERS THAT EXECUTE ORDERS IN NON-TRIVIAL AMOUNTS
class ExecutionalTrader(MarketAgent):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)
        self.intendedOrders = {}
        self.conditionalOrders = {}

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
    
    def removeOldIntendedOrders(self):
        self.intendedOrders.clear()
    
    def updatePositioning(self, market):
        self.partialExecuteMarket(markets[market])
    
    def partialExecuteMarket(self, orderBook):
        if orderBook in self.intendedOrders:
            order = self.intendedOrders[orderBook]
            if order["quantity"] > 0:
                if random.random() < 0.1:
                    # Determine a random amount to fill, between 0.05 and 0.01 x quantity
                    quantity_to_fill = max(1, int(random.uniform(0.01, 0.05) * order["quantity"]))
            
                    # Place the order
                    self.placeOrder(orderBook, order["direction"], orderBook.lastPrice, quantity_to_fill, "market")

                    # Update the remaining quantity
                    order["quantity"] -= quantity_to_fill
                    
                    # If the order is completely filled, remove it from intended orders
                    if order["quantity"] <= 0:
                        del self.intendedOrders[orderBook]

    def updateOrdersInLegs(self, orderBook):
        if orderBook.bestBid is None or orderBook.bestAsk is None:
            return
        try:
            if self.intendedOrders[orderBook]["direction"] == "buy" and self.intendedOrders[orderBook]["price"] >= orderBook.bestAsk.getPrice():
                quantityToFill = min(self.intendedOrders[orderBook]["quantity"],orderBook.bestAsk.getQuantity())
                self.placeOrder(orderBook, "buy", 0, quantityToFill, "market")
                self.intendedOrders[orderBook]["quantity"] -= quantityToFill
            elif self.intendedOrders[orderBook]["direction"] == "sell" and self.intendedOrders[orderBook]["price"] <= orderBook.bestBid.getPrice():
                quantityToFill = min(self.intendedOrders[orderBook]["quantity"],orderBook.bestBid.getQuantity())
                self.placeOrder(orderBook, "sell", 0, quantityToFill, "market")
                self.intendedOrders[orderBook]["quantity"] -= quantityToFill
        except:
            return 
