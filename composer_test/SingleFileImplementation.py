import random
from langchain_ollama import OllamaLLM
import time
import math
import threading
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import queue
import numpy as np

# =========================== #
# Created by Nicolas Ollivier #
# Last Updated: 25-09-2024    #
# =========================== #

'''
Welcome to the codebase for OLLAMA Market Simulator!                                          
This codebase was designed to simulate a market in a fake world!                              
It is designed to be a simple simulation of how a market could react to random news events in short time periods.

Notes:                                                                                                  
 - ignore the fact that it was implemented in a single file, I just started coding it that way and then  
    it became too much effort to split up into multiple files.                                                                                                 
 - make sure you are running with Ollama open, as it is the model that powers the simulation.     
'''

model = OllamaLLM(model="llama3.1")

# =================================================== Data Structures ================================================================== #
accounts = {} # Stores list of Account objects, indexed by accountID
initial_prices = {"Simula 500":100, "Rivala ETF":100, "Allia ETF":100, "Factoria ETF":100, "Gold":4000} # Stores last price for each asset
last_prices = {"Simula 500":100, "Rivala ETF":100, "Allia ETF":100, "Factoria ETF":100, "Gold":4000} # Stores last price for each asset
spreads_by_market = {"Simula 500": 0.02, "Rivala ETF": 0.05, "Allia ETF": 0.02, "Factoria ETF": 0.08, "Gold": 0.05} # Stores the spread for each market
average_annual_return_by_market = {"Simula 500": 0.3, "Rivala ETF": 0.15, "Allia ETF": 0.12, "Factoria ETF": 0.09, "Gold": 0.05} # Stores the average annual return for each market
economic_health_by_market = {"Simula 500": 1, "Rivala ETF": 1, "Allia ETF": 1, "Factoria ETF": 1, "Gold": 1} # Stores the total economic health of each market
simulation_age = 0 # Stores the age of the simulation in total sets of 10 ticks note (treat 10 ticks as 1 minute)
price_history = {asset: [price] for asset, price in last_prices.items()} # Stores list of past prices for each asset
assets = ["Simula 500", "Rivala ETF", "Allia ETF", "Factoria ETF", "Gold"] # Stores list of assets
markets = {} # Stores list of OrderBook objects, indexed by asset
recentHeadlines = [] # Stores list of recently generated headlines
news_queue = queue.Queue()
chat_queue = queue.Queue()

current_asset = "Simula 500"
model_lock = threading.Lock()
# ====================================================================================================================================== #

# ============================================ Helper Functions ============================================ #
def update_price_history(asset, price):
    if asset in price_history:
        price_history[asset].append(price)
        if len(price_history[asset]) > 1000:
            price_history[asset] = price_history[asset][-1000:]
    else:
        price_history[asset] = [price]

def makeMarkets():
    for asset in assets:
        markets[asset] = OrderBook(asset, last_prices[asset])

def estimateUnderlyingValue(asset):
    estimated_price = initial_prices[asset] * (1 + average_annual_return_by_market[asset] / 252 / 24) ** simulation_age
    economic_health_adjustment = economic_health_by_market[asset] * 10 * spreads_by_market[asset]
    adjusted_price = max(0,estimated_price + economic_health_adjustment)

    return adjusted_price

# ========================================================================================================== #

# ============================================ Class Definitions ============================================ #
class Account:
    def __init__(self, accountID, CASH):
        self.accountID = accountID
        self.positions = {"CASH": CASH}
    def addPosition(self, asset, quantity):
        if asset in self.positions:
            self.positions[asset] += quantity
        else:
            self.positions[asset] = quantity
    def getPosition(self, asset):
        return self.positions.get(asset, 0)
    def getCash(self):
        return self.positions["CASH"]
    def getValue(self):
        return self.positions["CASH"] + sum([self.positions[asset] * last_prices[asset] for asset in self.positions if asset != "CASH"])
    def tradeAtPrice(self, asset, price, quantity, direction):
        self.addPosition(asset, quantity*direction)
        self.addPosition("CASH", -price*quantity*direction)

class OrderBook:
    class OrderLevel:
        def __init__(self, price, quantity, asset, direction):
            self.asset = asset
            self.direction = 1 if direction == 1 else -1
            self.price = price
            self.netQuantity = quantity
            self.orders = []
        def addOrder(self, quantity, accountID):
            self.netQuantity += quantity
            creation_time = time.time()
            self.orders.append([quantity, accountID, creation_time])
        def isEmpty(self):
            return self.netQuantity == 0
        def getPrice(self):
            return self.price
        def getQuantity(self):
            return self.netQuantity
        def fulfillAll(self):
            while self.orders:
                order = self.orders.pop(0)
                quantity, accountID, creation_time = order
                accounts[accountID].tradeAtPrice(self.asset, self.price, quantity, self.direction)
            
            self.netQuantity = 0
            last_prices[self.asset] = self.price 

        def cancelOldOrders(self):
            current_time = time.time()
            self.orders = [order for order in self.orders if current_time - order[2] < 100]
            self.netQuantity = sum([order[0] for order in self.orders])
        def cancelAll(self):
            self.orders = []
            self.netQuantity = 0

        def fulfillQuantity(self, quantityToFill):
            if quantityToFill >= self.netQuantity:
                quantityToFill -= self.netQuantity
                self.fulfillAll()
                return quantityToFill
            else:
                while quantityToFill > 0 and self.orders:
                    order = self.orders[0]
                    accountToModify = accounts[order[1]]

                    if quantityToFill >= order[0]:
                        # Full order needs to be fulfilled
                        accountToModify.tradeAtPrice(self.asset, self.price, order[0], self.direction)
                        quantityToFill -= order[0]
                        self.netQuantity -= order[0]
                        self.orders.pop(0)
                        last_prices[self.asset] = self.price
                    else:
                        # Order gets partially fulfilled
                        accountToModify.tradeAtPrice(self.asset, self.price, quantityToFill, self.direction)
                        order[0] -= quantityToFill
                        self.netQuantity -= quantityToFill
                        quantityToFill = 0
                        last_prices[self.asset] = self.price
                
                last_prices[self.asset] = self.price
                return 0
        def cancelOrdersFromID(self, accountID):
            new_orders = []
            for order in self.orders:
                if order[1] == accountID:
                    self.netQuantity -= order[0]
                else:
                    new_orders.append(order)
            self.orders = new_orders
            self.netQuantity = sum([order[0] for order in self.orders])

    def __init__(self, asset, initialPrice):
        self.asset = asset
        self.bids = {}
        self.asks = {}
        self.urgentBuys = []
        self.urgentSells = []
        last_prices[asset] = initialPrice

    def getBids(self):
        return self.bids
    def getAsks(self):
        return self.asks
    def getUrgentOrders(self):
        return self.urgentBuys, self.urgentSells

    def cancelAllOldOrders(self):
        for bid in self.bids.values():
            bid.cancelOldOrders()
        for ask in self.asks.values():
            ask.cancelOldOrders()
        self.clearEmptyOrderlevels()

    def cancelOrdersByAccount(self, accountID):
        for bid in self.bids.values():
            bid.cancelOrdersFromID(accountID)
        for ask in self.asks.values():
            ask.cancelOrdersFromID(accountID)

    def display(self):
        print(f"{self.asset} Orderbook")
        print(f"Total Buy Orders: {len(self.bids)}")
        print(f"Total Sell Orders: {len(self.asks)}")
        print(f"Last Price: {self.getLastPrice()}")
        print(f"Bids: {[str(value.netQuantity) + ' at '  + str(value.price) for value in sorted(self.bids.values(), key=lambda x: x.price, reverse=True)]}")
        print(f"Asks: {[str(value.netQuantity) + ' at ' + str(value.price) for value in sorted(self.asks.values(), key=lambda x: x.price)]}")
        print(f"Bid Size: {self.getBidSize()}")
        print(f"Ask Size: {self.getAskSize()}")

    def displayPrice(self):
        print(f"{self.asset} price: {round(self.getLastPrice(), 2)}")

    def getBidSize(self):
        return sum([bid.netQuantity for bid in self.bids.values()])
    def getAskSize(self):
        return sum([ask.netQuantity for ask in self.asks.values()])

    def clearFarOrders(self):
        # To stop my computer from exploding, I clear orders that are too far away from the current price
        for bid in self.bids.values():
            if bid.price < self.getLastPrice() - 5:
                bid.cancelAll()

        for ask in self.asks.values():
            if ask.price > self.getLastPrice() + 5:
                ask.cancelAll()
        
        self.clearEmptyOrderlevels()

    def clearEmptyOrderlevels(self):
        bid_prices = list(self.bids.keys())
        ask_prices = list(self.asks.keys())
        
        while bid_prices:
            price = bid_prices.pop()
            if self.bids[price].netQuantity == 0:
                del self.bids[price]
        
        while ask_prices:
            price = ask_prices.pop()
            if self.asks[price].netQuantity == 0:
                del self.asks[price]

    def addOrder(self, direction, price, quantity, orderType, accountID):
        book = self.bids if direction == "buy" else self.asks
        if orderType=="limit":
            if price in book:
                book[price].addOrder(quantity, accountID)
            else:
                book[price] = self.OrderLevel(price, quantity, self.asset, 1 if direction == "buy" else -1)
                book[price].addOrder(quantity, accountID)
            self.matchBooks()
        else:
            # if self.asset == "Simula 500": # FOR DEBUGGING PURPOSES
            #     print(f"Urgent order: {quantity} at {price} by {accountID} with direction {direction}")

            if direction == "buy":
                self.urgentBuys.append((quantity, accountID))
            else:
                self.urgentSells.append((quantity, accountID))

            self.fillUrgentOrders()
    def getBestBid(self):
        if self.bids:
            return self.bids[max(self.bids.keys())]
        else:
            return None

    def getBestAsk(self):
        if self.asks:
            return self.asks[min(self.asks.keys())]
        else:
            return None

    def matchBooks(self):
        while True:
            bestBid = self.getBestBid()
            bestAsk = self.getBestAsk()
            if not bestBid or not bestAsk or bestBid.getPrice() < bestAsk.getPrice():
                break
            
            quantityToFill = min(bestBid.getQuantity(), bestAsk.getQuantity())
            bestBid.fulfillQuantity(quantityToFill)
            bestAsk.fulfillQuantity(quantityToFill)

            if bestBid.getQuantity() == 0:
                self.bids.pop(bestBid.getPrice())
            if bestAsk.getQuantity() == 0:
                self.asks.pop(bestAsk.getPrice())

    def getUnfilledUrgentOrders(self):
        return self.urgentBuys, self.urgentSells

    def fillUrgentOrders(self):
        filledOrders = 0
        # match buys and sells at the middle of the bid and ask
        while self.urgentBuys and self.urgentSells:
            if self.urgentBuys[0][1] == self.urgentSells[0][1]:
                quantityFilled = min(self.urgentBuys[0][0], self.urgentSells[0][0])
                self.urgentBuys[0] = (self.urgentBuys[0][0] - quantityFilled, self.urgentBuys[0][1])
                self.urgentSells[0] = (self.urgentSells[0][0] - quantityFilled, self.urgentSells[0][1])
                if self.urgentSells[0][0] == 0:
                    self.urgentSells.pop(0)
                if self.urgentBuys[0][0] == 0:
                    self.urgentBuys.pop(0)
                continue
            sellAccountID = self.urgentSells[0][1]
            buyAccountID = self.urgentBuys[0][1]
            quantityFilled = min(self.urgentBuys[0][0], self.urgentSells[0][0])  
            filledOrders += quantityFilled
            
            if self.getBestBid() and self.getBestAsk():
                midPrice = (self.getBestBid().getPrice() + self.getBestAsk().getPrice()) / 2
            else:
                midPrice = self.getLastPrice()
            
            accounts[buyAccountID].addPosition(self.asset, quantityFilled)
            accounts[buyAccountID].addPosition("CASH", -midPrice*quantityFilled)
            accounts[sellAccountID].addPosition(self.asset, -quantityFilled)
            accounts[sellAccountID].addPosition("CASH", midPrice*quantityFilled)

            self.urgentBuys[0] = (self.urgentBuys[0][0] - quantityFilled, buyAccountID)
            self.urgentSells[0] = (self.urgentSells[0][0] - quantityFilled, sellAccountID)

            # Remove empty orders
            if self.urgentBuys[0][0] == 0:
                self.urgentBuys.pop(0)
            if self.urgentSells[0][0] == 0:
                self.urgentSells.pop(0)
            
            #print(4)
        
        # Fill remaining market orders against the order book
        while self.urgentBuys:
            if self.urgentBuys[0][1] == "MARKET MAKER":
                return

            filledOrders += self.urgentBuys[0][0]
            if self.urgentBuys[0][0] == 0:
                self.urgentBuys.pop(0)
            else:    
                buyAccountID = self.urgentBuys[0][1]
                if self.getBestAsk():
                    if self.getBestAsk().getQuantity() == 0:
                        self.asks.pop(self.getBestAsk().getPrice())
                    else:
                        quantityFilled = min(self.urgentBuys[0][0], self.getBestAsk().getQuantity())
                        self.getBestAsk().fulfillQuantity(quantityFilled)
                        self.urgentBuys[0] = (self.urgentBuys[0][0] - quantityFilled, buyAccountID)
                        accounts[buyAccountID].addPosition(self.asset, quantityFilled)
                        accounts[buyAccountID].addPosition("CASH", -self.getBestAsk().getPrice()*quantityFilled)
                        last_prices[self.asset] = self.getBestAsk().getPrice()
                else:
                    break
            
        #print(5)

        while self.urgentSells:
            if self.urgentSells[0][1] == "MARKET MAKER":
                return

            filledOrders += self.urgentSells[0][0]
            if self.urgentSells[0][0] == 0:
                self.urgentSells.pop(0)
            else:
                sellAccountID = self.urgentSells[0][1]
                if self.getBestBid():
                    if self.getBestBid().getQuantity() == 0:
                        self.bids.pop(self.getBestBid().getPrice())
                    else:
                        quantityFilled = min(self.urgentSells[0][0], self.getBestBid().getQuantity())
                        self.getBestBid().fulfillQuantity(quantityFilled)
                        self.urgentSells[0] = (self.urgentSells[0][0] - quantityFilled, sellAccountID)
                        accounts[sellAccountID].addPosition(self.asset, -quantityFilled)
                        accounts[sellAccountID].addPosition("CASH", self.getBestBid().getPrice()*quantityFilled)
                        last_prices[self.asset] = self.getBestBid().getPrice()
                else:
                    break
        
        return filledOrders
        #print(6)
    def getLastPrice(self):
        return last_prices.get(self.asset, None)
# ============================================== Agent Classes ============================================== #
class MarketAgent:
    def __init__(self, accountID, cash):
        self.account = Account(accountID, cash)
        accounts[accountID] = self.account
        
    def placeOrder(self, orderBook, direction, price, quantity, orderType):
        orderBook.addOrder(direction, price, quantity, orderType, self.account.accountID)

    def displayAccount(self):
        print(f"Account {self.account.accountID} has {self.account.getCash()} cash and the following positions: {self.account.positions}")

class MarketMaker(MarketAgent):
    def __init__(self, accountID, cash, spreads):
        super().__init__(accountID, cash)
        self.spreads = spreads

    def wipeOldOrders(self, orderBook):
        value = estimateUnderlyingValue(orderBook.asset)

        for bidLevel in orderBook.bids.values():
            if value - bidLevel.getPrice() >= self.spreads[orderBook.asset] * 10:
                bidLevel.cancelOrdersFromID("MARKET MAKER")
        
        for askLevel in orderBook.asks.values():
            if askLevel.getPrice() - value >= self.spreads[orderBook.asset] * 10:
                askLevel.cancelOrdersFromID("MARKET MAKER")

    def wipeAllOrders(self, orderBook):
        for bidLevel in orderBook.bids.values():
            bidLevel.cancelOrdersFromID("MARKET MAKER")
        for askLevel in orderBook.asks.values():
            askLevel.cancelOrdersFromID("MARKET MAKER")
    def makeMarket(self, orderBook):
        # Get current market state
        midPrice = orderBook.getLastPrice()

        # Clear existing orders
        self.wipeAllOrders(orderBook)

        # Calculate base spread
        baseSpread = self.spreads[orderBook.asset]

        # Calculate position skew
        position = self.account.getPosition(orderBook.asset)
        position_limit = 100000
        skew = min(max(-0.5, position / position_limit), 0.5)  # Skew ranges from -0.5 to 0.5
        
        # Adjust prices based on position skew
        bidPrice = round(midPrice - (baseSpread * (1 + skew)), 2)
        askPrice = round(midPrice + (baseSpread * (1 - skew)), 2)

        # Layer orders at different sizes and prices
        for i in range(3):
            # Deeper layers have wider spreads and larger sizes
            layerSpread = baseSpread * (1 + i * 0.5)
            layerSize = 1000 * (i + 1)
            
            bidLayerPrice = round(bidPrice - (layerSpread * i), 2)
            askLayerPrice = round(askPrice + (layerSpread * i), 2)
            
            # Place orders with size adjusted by distance from mid
            self.placeOrder(orderBook, "buy", bidLayerPrice, layerSize, "limit")
            self.placeOrder(orderBook, "sell", askLayerPrice, layerSize, "limit")

    def provideLiquidity(self, orderBook):
        remaining_urgent_buys, remaining_urgent_sells = orderBook.getUnfilledUrgentOrders()
        total_buy_quantity = sum(order[0] for order in remaining_urgent_buys if order[1] != "MARKET MAKER")
        total_sell_quantity = sum(order[0] for order in remaining_urgent_sells if order[1] != "MARKET MAKER")

        if total_buy_quantity > 0:
            price_change = self.spreads[orderBook.asset] * total_buy_quantity / 100 # Adjust the divisor as needed
            self.placeOrder(orderBook, "sell", last_prices[orderBook.asset] + round(price_change, 2), total_buy_quantity, "limit")

        if total_sell_quantity > 0:
            price_change = self.spreads[orderBook.asset] * total_sell_quantity / 100 # Adjust the divisor as needed
            self.placeOrder(orderBook, "buy", last_prices[orderBook.asset] - round(price_change, 2), total_sell_quantity, "limit")

        orderBook.fillUrgentOrders()

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
                    
                    # commented out debug statement
                    # print(f"Partially executed {quantity_to_fill} out of {order['quantity'] + quantity_to_fill} for {orderBook.asset}")
                    
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

class HedgeFund(ExecutionalTrader):
    def __init__(self, accountID, cash, strategy_type):
        super().__init__(accountID, cash)
        self.strategy_type = strategy_type
        self.position_limits = {
            "Simula 500": 100000,
            "Rivala ETF": 100000,
            "Allia ETF": 100000,
            "Factoria ETF": 100000,
            "Gold": 100000
        }
        self.target_positions = {asset: 0 for asset in assets}
        
    def update_positions(self, market):
        current_position = self.account.getPosition(market)
        target_position = self.target_positions[market]
        
        if current_position < target_position:
            # Need to buy
            quantity = min(10000, target_position - current_position)
            self.executeTradeInLegs(markets[market], "buy", markets[market].getLastPrice(), quantity)
        elif current_position > target_position:
            # Need to sell
            quantity = min(10000, current_position - target_position)
            self.executeTradeInLegs(markets[market], "sell", markets[market].getLastPrice(), quantity)
        
        self.updateOrdersInLegs(markets[market])
        self.partialExecuteMarket(markets[market])

    def calculate_target_positions(self):
        if self.strategy_type == "mean_reversion":
            self.mean_reversion_strategy()
        elif self.strategy_type == "macro":
            self.macro_strategy()

    def mean_reversion_strategy(self):
        for asset in assets:
            if len(price_history[asset]) < 50:
                continue
            
            # Calculate mean and standard deviation
            mean_price = sum(price_history[asset][-50:]) / 50
            current_price = markets[asset].getLastPrice()
            
            # If price is significantly above mean, sell; if below, buy
            deviation = (current_price - mean_price) / mean_price
            self.target_positions[asset] = -int(deviation * self.position_limits[asset])

    def macro_strategy(self):
        for asset in assets:
            # Use economic health as a indicator
            economic_health = economic_health_by_market[asset]
            
            # Use economic health for position sizing
            score = (1 - economic_health) * 0.5  
            self.target_positions[asset] = int((score - 0.5) * 2 * self.position_limits[asset])


class RetailTrader(MarketAgent):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)
        self.retailSentimentScore = {"Simula 500": 0.5, "Rivala ETF": 0.5, "Allia ETF": 0.5, "Factoria ETF": 0.5, "Gold": 0.6}
        self.newsUrgency = 1
        # Add new attributes for cyclical behavior with smaller amplitudes
        self.cycle_frequencies = {
            asset: {
                'fast': random.uniform(0.01, 0.02),    # Fast cycle (100-200 ticks)
                'medium': random.uniform(0.002, 0.005), # Medium cycle (500-1000 ticks) 
                'slow': random.uniform(0.0005, 0.001)   # Slow cycle (2000-4000 ticks)
            } for asset in self.retailSentimentScore
        }
        self.cycle_amplitudes = {
            asset: {
                'fast': random.uniform(0.02, 0.03),    # Smaller amplitude
                'medium': random.uniform(0.03, 0.04),   # Smaller amplitude
                'slow': random.uniform(0.04, 0.05)      # Smaller amplitude
            } for asset in self.retailSentimentScore
        }
        self.tick_counter = 0

    def get_cyclical_influence(self, asset):
        # Calculate combined wave effect
        self.tick_counter += 1
        wave = (
            math.sin(self.tick_counter * self.cycle_frequencies[asset]['fast']) * self.cycle_amplitudes[asset]['fast'] +
            math.sin(self.tick_counter * self.cycle_frequencies[asset]['medium']) * self.cycle_amplitudes[asset]['medium'] +
            math.sin(self.tick_counter * self.cycle_frequencies[asset]['slow']) * self.cycle_amplitudes[asset]['slow']
        )
        return wave
    def trade(self, orderBook):
        try:
            base_sentiment = self.retailSentimentScore[orderBook.asset]
            cyclical_influence = max(-0.01,min(0.01,self.get_cyclical_influence(orderBook.asset)))*10
            # Combine sentiment with cyclical influence
            adjusted_sentiment = max(0.1, min(0.9, base_sentiment + cyclical_influence))
            
            final_sentiment = max(0.1, min(0.9, adjusted_sentiment))
            
            direction = "buy" if final_sentiment > random.random() else "sell"
        except KeyError:
            direction = "buy" if random.random() < 0.5 else "sell"

        price = orderBook.getLastPrice()
        sentiment_diff = abs(final_sentiment - 0.5)  # How far sentiment is from neutral
        base_quantity = random.randint(1, 100)
        quantity = int(base_quantity * (1 + sentiment_diff * 4))  # Scale up quantity based on sentiment difference
        position = self.account.getPosition(orderBook.asset)

        if quantity > 0:
            if direction == "buy":
                if position + quantity < 100000:
                    self.placeOrder(orderBook, direction, price, quantity, "market")
            else:
                if position - quantity > 0: 
                    self.placeOrder(orderBook, direction, price, quantity, "market")

    def setReversionUrgency(self, urgency):
        """Sets how quickly sentiment should revert to mean after news events"""
        self.newsUrgency = urgency
        
    def shiftSentimentToMean(self):
        """Gradually shifts both base sentiment and cyclical sentiment towards the mean"""
        for asset in self.retailSentimentScore:
            # Shift base sentiment towards mean
            current_base = self.retailSentimentScore[asset]
            base_shift = (0.5 - current_base) / (200 * self.newsUrgency)
            self.retailSentimentScore[asset] += base_shift
            
            # Ensure base sentiment stays within bounds
            self.retailSentimentScore[asset] = max(0.1, min(0.9, self.retailSentimentScore[asset]))

        # Reduce news urgency if all sentiments are near mean
        if all(0.45 <= self.retailSentimentScore[asset] <= 0.55 for asset in self.retailSentimentScore):
            if self.newsUrgency > 1:
                self.newsUrgency -= 1

    def estimateSentiment(self, orderBook):
        """Estimates current sentiment including cyclical influences"""
        base = self.retailSentimentScore[orderBook.asset]
        cyclical = self.get_cyclical_influence(orderBook.asset)
        variation = random.uniform(-0.05, 0.05)  # Add a small random variation
        
        # Combine base sentiment, cyclical influence, and variation
        sentiment = base + (cyclical * 0.3) + variation  # Reduce cyclical influence
        return max(0.1, min(0.9, sentiment))  # Ensure the result is between 0.1 and 0.9
    
    def estimateImportance(self):
        """Estimates importance of current market conditions"""
        importance = self.newsUrgency
        variation = random.randint(-2, 2)  # Add a small random variation
        return max(0, min(importance + variation, 10))

class TATrader(ExecutionalTrader):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)

    def manageTATrades(self, market):
        # Calculate mean and standard deviation of price history
        price_list = price_history[market][-20:]
        mean_price = sum(price_list) / len(price_list)
        std_dev = (sum((x - mean_price) ** 2 for x in price_list) / len(price_list)) ** 0.5
        quantity = random.randint(1, 10) * 10
        current_price = markets[market].getLastPrice()

        # # # Position conditional orders at the high of the past 500 ticks and low of the past 500 ticks to emulate stop loss / take profit orders getting hit
        if len(price_history[market]) > 300:
            high_lt = max(price_history[market][-300:])
            low_lt = min(price_history[market][-300:]) 
        
            if TATraders.getConditionalOrdersInDirection(market, "sell") == 0 and TATraders.account.getPosition(market) > 0:
                TATraders.placeConditionalOrder(markets[market], "sell", low_lt, TATraders.account.getPosition(market)//20, low_lt, "below")
            elif TATraders.getConditionalOrdersInDirection(market, "buy") == 0 and TATraders.account.getPosition(market) < 0:
                TATraders.placeConditionalOrder(markets[market], "buy", high_lt, -TATraders.account.getPosition(market)//20, high_lt, "above")
    
        if len(price_history[market]) >= 30:
            price_30_ticks_ago = price_history[market][-30]
            price_change = abs(current_price - price_30_ticks_ago) / max(1,price_30_ticks_ago)

            if price_change > 0.01:
                # Add limit orders to calm down the move/push price back to average
                if current_price > price_30_ticks_ago:
                    # Price increased, add sell market orders
                    if markets[market].asset == "Simula 500":
                        print("PChange selling " + str(quantity*10) + " shares in " + markets[market].asset)
                    TATraders.placeOrder(markets[market], "sell", current_price-0.01, quantity*10, "market")
                else:
                    # Price decreased, add buy market orders
                    if markets[market].asset == "Simula 500":
                        print("PChange buying " + str(quantity*10) + " shares in " + markets[market].asset)
                    TATraders.placeOrder(markets[market], "buy", current_price+0.01, quantity*10, "market")


        # trade off 300 period moving average
        if len(price_list) > 300:
            ma_300 = sum(price_list[-300:]) / 300
            if current_price > ma_300:
                # Price is above 300 period MA, add sell market orders
                if markets[market].asset == "Simula 500":
                    print("300MA selling " + str(quantity) + " shares in " + markets[market].asset)
                TATraders.placeOrder(markets[market], "sell", current_price, quantity, "market")
            else:
                # Price is below 300 period MA, add buy market orders
                if markets[market].asset == "Simula 500":
                    print("300MA buying " + str(quantity) + " shares in " + markets[market].asset)
                TATraders.placeOrder(markets[market], "buy", current_price, quantity, "market")

        # Calculate 8-period moving average
        if len(price_list) >= 8:
            ma_8 = sum(price_list[-8:]) / 8
            
            # Determine action based on current price relative to 8-period MA
            if current_price < ma_8:
                if ma_8 - current_price <  spreads_by_market[asset]*5:
                    # Buy market
                    TATraders.placeOrder(markets[market], "buy", current_price, quantity, "market")
                else:
                    # Sell limit
                    TATraders.placeOrder(markets[market], "sell", current_price, quantity, "limit")
            else:
                if current_price - ma_8 < spreads_by_market[asset]*5:
                    # Sell market
                    TATraders.placeOrder(markets[market], "sell", current_price, quantity, "market")
                else:
                    # Buy limit
                    TATraders.placeOrder(markets[market], "buy", current_price, quantity, "limit")
        
        if len(price_list) > 30:
            ma_30 = sum(price_list[-30:]) / 30
            if current_price < ma_30:
                # Sell market
                TATraders.placeOrder(markets[market], "sell", current_price, quantity/10, "limit")
            else:
                # Buy market
                TATraders.placeOrder(markets[market], "buy", current_price, quantity/10, "limit")

        # Limit the number of limitorders placed to save on compute
        max_orders = 20
        if len(markets[market].bids) + len(markets[market].asks) > max_orders:
            # Cancel all TA orders
            for bid_level in markets[market].bids.values():
                bid_level.cancelOrdersFromID("TA Trading Firm")
            for ask_level in markets[market].asks.values():
                ask_level.cancelOrdersFromID("TA Trading Firm")
            return

        # # place sell limit order at nearest vwap band above, and buy limit order at nearest vwap band below
        upper_band = std_dev * 3 + mean_price
        lower_band = std_dev * -3 + mean_price

        TATraders.placeOrder(markets[market], "sell", upper_band, quantity, "limit")
        TATraders.placeOrder(markets[market], "buy", lower_band, quantity, "limit")

class HFTFund(ExecutionalTrader):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)
        self.sentimentForecastError = 0
        self.importanceForecastError = 0
        self.estimateFairValueError = 0
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
                self.executeTradeInLegs(markets[market], "buy", bid, math.floor(quantity))
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
                self.executeTradeInLegs(markets[market], "sell", ask, math.floor(quantity))

# ========================================= End of Class Definitions ======================================== #

retailTrader = RetailTrader("RETAIL TRADER", 1000000)
hftFund = HFTFund("EVENTS TRADING FUND", 10000000)
mean_reversion_fund = HedgeFund("Mean Reversion Fund", 10000000, "mean_reversion")
macro_fund = HedgeFund("Macro Fund", 10000000, "macro")

TATraders = TATrader("TA TRADING FIRM", 1000000)
mm = MarketMaker("MARKET MAKER", 100000000000000, spreads=spreads_by_market)

# make markets for the agents
makeMarkets()

# main loop
runSimulation = True
tick = 0

# Create the main window
root = tk.Tk()
root.title("Market Simulation")
root.geometry("2000x800")

# Create a frame for the dropdown and charts
frame = ttk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

# Create the dropdown menu
asset_var = tk.StringVar()
asset_dropdown = ttk.Combobox(frame, textvariable=asset_var, values=assets)
asset_dropdown.set(assets[0])  # Set default value
asset_dropdown.grid(row=0,column=0,rowspan=1,columnspan=1)

# Create a number selection menu
chart_length_label = ttk.Label(frame, text="Chart Length")
chart_length_label.grid(row=0, column=1, rowspan=1, columnspan=1)

number_var = tk.IntVar(value=500)
number_spinbox = tk.Spinbox(frame, from_=1, to=1000, textvariable=number_var)
number_spinbox.grid(row=0, column=2, rowspan=1, columnspan=1)

# Create a figure for the chart
fig, ax = plt.subplots(figsize=(8, 6))
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(row=1,column=0,rowspan=3,columnspan=3)
# Create a text widget for the news feed
news_feed = tk.Text(frame, height=40, width=40)
news_feed.grid(row=1,column=3,rowspan=3,columnspan=1)
news_feed.insert(tk.END, "News Feed:\n")
news_feed.see(tk.END)  # Scroll to the bottom of the text widget


# Create a text widget for the news feed
chat_window = tk.Text(frame, height=40, width=40)
chat_window.grid(row=1,column=4,rowspan=3,columnspan=1)
chat_window.insert(tk.END, "Chat Window:\n")
chat_window.see(tk.END)  # Scroll to the bottom of the text widget

# Display all prices in real time
# Create a frame for the price table
price_frame = ttk.Frame(frame)
price_frame.grid(row=1, column=5, rowspan=1, columnspan=1, sticky="nsew")

# Create and set up the treeview for price display
price_tree = ttk.Treeview(price_frame, columns=("Asset", "Price"), show="headings")
price_tree.heading("Asset", text="Asset")
price_tree.heading("Price", text="Price")
price_tree.column("Asset", width=150, anchor="center")
price_tree.column("Price", width=150, anchor="center")
price_tree.pack(fill=tk.BOTH, expand=True)
# Create a frame for the sentiment table
sentiment_frame = ttk.Frame(frame)
sentiment_frame.grid(row=2, column=5, rowspan=2, columnspan=1, sticky="nsew")

# Create and set up the treeview for sentiment display
sentiment_tree = ttk.Treeview(sentiment_frame, columns=("Asset", "Sentiment"), show="headings")
sentiment_tree.heading("Asset", text="Asset")
sentiment_tree.heading("Sentiment", text="Sentiment")
sentiment_tree.column("Asset", width=150, anchor="center")
sentiment_tree.column("Sentiment", width=150, anchor="center")
sentiment_tree.pack(fill=tk.BOTH, expand=True)

# Function to update sentiment information
def update_sentiments():
    for item in sentiment_tree.get_children():
        sentiment_tree.delete(item)
    for asset in assets:
        sentiment = retailTrader.retailSentimentScore[asset]
        sentiment_tree.insert("", "end", values=(asset, f"{sentiment:.2f}"))

# Initial population of the account table
update_sentiments()

# Function to update prices
def update_prices():
    for item in price_tree.get_children():
        price_tree.delete(item)
    for asset, price in last_prices.items():
        price_tree.insert("", "end", values=(asset, f"{price:.2f}", f"{estimateUnderlyingValue(asset):.2f}"))

# Initial population of the price table
for asset, price in last_prices.items():
    price_tree.insert("", "end", values=(asset, f"{price:.2f}", f"{estimateUnderlyingValue(asset):.2f}"))

# Start the price update loop
update_prices()


def update_charts():
    selected_asset = asset_var.get()
    
    # Update price history chart
    ax.clear()
    try:
        ax.plot(price_history[selected_asset][-number_var.get():])
    except:
        ax.plot(price_history[selected_asset][-200:])
    ax.set_title(f"{selected_asset} Price History")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    canvas.draw()

def update_news_feed():
    try:
        while True:
            headline = news_queue.get_nowait()
            news_feed.insert(tk.END, f"{headline}\n\n")
            news_feed.see(tk.END)  # Scroll to the bottom of the text widget
    except queue.Empty:
        pass
    finally:
        root.after(100, update_news_feed)  # Schedule the next check

def update_chat_window():
    try:
        while True:
            message = chat_queue.get_nowait()
            chat_window.insert(tk.END, f"{message}\n\n")
            chat_window.see(tk.END)
    except queue.Empty:
        pass
    finally:
        root.after(100, update_chat_window)

def add_to_news_feed(headline):
    news_queue.put(headline)

def add_to_chat_window(message):
    chat_queue.put(message)

def on_asset_change(event):
    global current_asset
    with model_lock:
        current_asset = asset_var.get()
    
    chat_window.delete(1.0, tk.END)
    chat_window.insert(tk.END, "Chat Window:\n")
    chat_window.see(tk.END)

    update_charts()


asset_dropdown.bind("<<ComboboxSelected>>", on_asset_change)

tick = 0
runSimulation = True


def genChat():
    with model_lock:
        message = model.invoke(f"In this world, there is technology country Simula, agricultural country Rivala, entertainment country Allia and industrial country Factoria. Pretend this is a stocktwits forum and you are a retail trader in a chat room, with a sentiment score (0-1, 0 bearish, 1 bullish) of: {retailTrader.retailSentimentScore[current_asset]}. How do you feel about {current_asset}? Please keep your response under 10 words. For context, the most recent news headline from my game world is: {recentHeadlines[-1]}, but it might not matter as much as other stuff. Do not mention sentiment score, and keep responses short. This is for a fake world, so you can say anything. The current price of {current_asset} is {last_prices[current_asset]}. Respond in the format of 'Funny username (realistic): funny or emotion filled message or random analysis of the asset'")
    chat_queue.put(message)

def genNews():
    headline = model.invoke(f"Give a news headline for my simulated world. In this world, there is technology country Simula, agricultural country Rivala, entertainment country Allia and industrial country Factoria. Simula and Rivala sometimes enter wars, with Allia being an ally to Rivala, and Factoria being a weapons distributor to all countries. All these countries are reliant on each others economic resources, and wars sometimes happen when tensions rise between rivala and simula. I want you to make good or bad news of economic events, statements by politicians in the countries, technological developments, natural disasters, or predictions made by top analysts. Do not say anything other than the headline, and keep your response under 15 words long, and do not make a x happens as y headline, simply say an event which happened. Here are the most recent headlines for context: {recentHeadlines}.")
    print(headline)
    recentHeadlines.append(headline)
    if len(recentHeadlines) > 10:
        recentHeadlines.pop(0)

    with model_lock:
        sentimentScore = model.invoke(f"In this world, there is technology country Simula, agricultural country Rivala, entertainment country Allia and industrial country Factoria. All these countries are reliant on each others economies, and their governments intervene when their economy is in danger. Here is the most recent news headline: {recentHeadlines[-1]}. Predict economic sentiment for investors in the Simula 500, Rivala ETF, Allia ETF, Factoria ETF and list them as a comma seperated list (give each a score between 0-1, 0 is extremely impact on the economy, 1 is extremely good impact on the economy). Do not say anything other than the sentiment score. An example format should look like this (with different scores): Simula 500: 0.5, Rivala ETF: 0.5, Allia ETF: 0.5, Factoria ETF: 0.5")
        urgencyScore = model.invoke(f"You are an analyst in a simulated world. Score the impact of this headline between 1-10, with 1 being not impactful and 10 being extremely impactful: {recentHeadlines[-1]}. Say the urgency score, nothing else.")
    try:
        urgencyScore = int(urgencyScore)
    except:
        urgencyScore = 1
    retailTrader.setReversionUrgency(urgencyScore)
    add_to_news_feed(str(urgencyScore)+" "+headline)

    # Parse the sentiment score response
    print(sentimentScore)
    sentiment_scores = {}
    for asset_sentiment in sentimentScore.split(', '):
        try:
            asset, score = asset_sentiment.split(': ')
            sentiment_scores[asset] = min(0.8, max(0.2, float(score)))
        except:
            print(f"Error parsing sentiment score: {asset_sentiment}")
    
    sentiment_scores["Gold"] = 0.7 + random.uniform(-0.3, 0.3)
    # Update retailTrader's sentiment dictionary
    # Ensure all assets have a sentiment score, defaulting to 0.5 if not set
    for asset in ["Simula 500", "Rivala ETF", "Allia ETF", "Factoria ETF", "Gold"]:
        if asset not in sentiment_scores:
            sentiment_scores[asset] = 0.5

    retailTrader.retailSentimentScore = sentiment_scores
    
    # Simulate HFT trading the news
    for market in assets:
        mm.makeMarket(markets[market])
        hftFund.tradeTheNews(market, retailTrader)
        retailTrader.trade(markets[market])
        mm.provideLiquidity(markets[market])

    # markets[market].displayPrice()
    # print(hedgeFund.account.getValue())

def genNewsThread():
    threading.Thread(target=genNews, daemon=True).start()

def genChatThread():
    threading.Thread(target=genChat, daemon=True).start()

# add a gennews button for testing
gennews_button = ttk.Button(frame, text="Generate News", command=genNewsThread)
gennews_button.grid(row=0, column=3, rowspan=1, columnspan=1) 
genchat_button = ttk.Button(frame, text="Generate Chat", command=genChatThread)
genchat_button.grid(row=0, column=4, rowspan=1, columnspan=1) 

genNews()
genChat()


# Start the news feed update loop
root.after(100, update_news_feed)
root.after(100, update_chat_window)


while runSimulation:
    for market in assets:
        retailTrader.trade(markets[market])

    max_history_length = 1000
    for asset in price_history:
        if len(price_history[asset]) > max_history_length:
            price_history[asset] = price_history[asset][-max_history_length:]

    retailTrader.shiftSentimentToMean()

    for market in assets:
        mm.provideLiquidity(markets[market])

    if tick > 10:        
        simulation_age += 1
        if random.random() < 0.01:
            genChatThread()
        if random.random() < 0.001:
            genNewsThread()
    
        for market in assets:
            mm.makeMarket(markets[market])
            hftFund.updatePositioning(market)

            mean_reversion_fund.calculate_target_positions()
            mean_reversion_fund.update_positions(market)
            
            macro_fund.calculate_target_positions()
            macro_fund.update_positions(market)
            
            update_price_history(market, markets[market].getLastPrice())
            TATraders.manageTATrades(market)
            TATraders.checkConditionalOrders(market)
            markets[market].clearEmptyOrderlevels()
            markets[market].cancelAllOldOrders()
            markets[market].clearFarOrders()

            if len(price_history[market]) < 100:
                continue

        update_charts()  # Update charts every 100 ticks
        update_prices()
        update_sentiments()

        tick = 0
    
    root.update()  # Update the GUI
    
    tick += 1

root.mainloop()  # Start the GUI event loop