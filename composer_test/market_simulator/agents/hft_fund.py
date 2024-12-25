import random
from market_simulator.agents.base_agent import MarketAgent
from market_simulator.utils.market_utils import estimateUnderlyingValue, markets

class HFTFund(MarketAgent):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)
        self.sentimentForecastError = 0
        self.importanceForecastError = 0
        self.estimateFairValueError = 0
        self.intendedOrders = {}
        print(self.sentimentForecastError, self.importanceForecastError, self.estimateFairValueError)

    def updatePositioning(self, market):
        self.updateOrdersInLegs(markets[market])
        self.partialExecuteMarket(markets[market])

    def estimateFairValue(self, market):
        return estimateUnderlyingValue(market) + self.estimateFairValueError
    
    def estimateImportance(self, rt):
        return max(0, min(10, rt.estimateImportance() + self.importanceForecastError))
    
    def estimateSentiment(self, rt, market):
        return max(0, min(1, rt.estimateSentiment(markets[market]) + self.sentimentForecastError))

    def tradeTheNews(self, market, rt):
        #  If sentiment is above 0.9 or below 0.1, make the HFT front run the trade by market buying/selling and then doing a smart execution to close
        if self.estimateSentiment(rt, market) <= 0.3:
            try:
                current_price = markets[market].getLastPrice()
                bid = markets[market].getBestBid().price
            except:
                current_price = markets[market].getLastPrice()
                bid = markets[market].getLastPrice()
            quantity = pow(self.estimateImportance(rt),2)*5
            if self.estimateImportance(rt) >= 8:
                self.placeOrder(markets[market], "sell", current_price, quantity, "market")
            else:
                self.placeOrder(markets[market], "sell", current_price, quantity, "market")
                print("selling " + str(quantity) + " shares in " + markets[market].asset)
                self.executeTradeInLegs(markets[market], "buy", bid, int(quantity))
        elif self.estimateSentiment(rt, market) >= 0.7:
            try:
                current_price = markets[market].getLastPrice()
                ask = markets[market].getBestAsk().price
            except:
                current_price = markets[market].getLastPrice()
                ask = markets[market].getLastPrice()
            quantity = pow(self.estimateImportance(rt),2)*5
            if self.estimateImportance(rt) >= 8:
                self.placeOrder(markets[market], "buy", current_price, quantity, "market")
            else:
                self.placeOrder(markets[market], "buy", current_price, quantity, "market")
                print("buying " + str(quantity) + " shares in " + markets[market].asset)
                self.executeTradeInLegs(markets[market], "sell", ask, int(quantity))

    def executeTradeInLegs(self, orderBook, direction, price, quantity):
        self.intendedOrders[orderBook] = {"direction": direction, "price": price, "quantity": quantity}
    
    def removeOldIntendedOrders(self):
        self.intendedOrders.clear()

    def partialExecuteMarket(self, orderBook):
        if random.random() < 0.05:
            if orderBook in self.intendedOrders:
                order = self.intendedOrders[orderBook]
                if order["quantity"] > 0:
                    # Determine a random amount to fill, between 0 and the full quantity
                    quantity_to_fill = random.randint(1, max(order["quantity"]//2,1))
                    
                    # Place the order
                    self.placeOrder(orderBook, order["direction"], order["price"], quantity_to_fill, "market")
                    
                    # Update the remaining quantity
                    order["quantity"] -= quantity_to_fill
                    
                    # If the order is completely filled, remove it from intended orders
                    if order["quantity"] <= 0:
                        del self.intendedOrders[orderBook]

    def updateOrdersInLegs(self, orderBook):
        if orderBook.getBestBid() is None or orderBook.getBestAsk() is None:
            return
        try:
            if self.intendedOrders[orderBook]["direction"] == "buy" and self.intendedOrders[orderBook]["price"] >= orderBook.getBestAsk().getPrice():
                quantityToFill = min(self.intendedOrders[orderBook]["quantity"],orderBook.getBestAsk().getQuantity())
                self.placeOrder(orderBook, "buy", 0, quantityToFill, "market")
                self.intendedOrders[orderBook]["quantity"] -= quantityToFill
            elif self.intendedOrders[orderBook]["direction"] == "sell" and self.intendedOrders[orderBook]["price"] <= orderBook.getBestBid().getPrice():
                quantityToFill = min(self.intendedOrders[orderBook]["quantity"],orderBook.getBestBid().getQuantity())
                self.placeOrder(orderBook, "sell", 0, quantityToFill, "market")
                self.intendedOrders[orderBook]["quantity"] -= quantityToFill
        except:
            return 