from market_simulator.agents.executional_trader import ExecutionalTrader
from market_simulator.utils.market_utils import markets, price_history, last_prices
from market_simulator.config import SPREADS, HFT_BASE_ORDER_SIZE, HFT_POSITION_LIMIT
import random

class HFTFund(ExecutionalTrader):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)
        self.intendedOrders = {}

    def tradeMicrostructure(self, ticker):
        self.cancelAllOrders(markets[ticker])
        posSize = self.account.getPosition(ticker)
        ob = markets[ticker]

        # front run big limit orders
        bidBook, askBook = ob.getBidAskPairs()
        for price, quantity in bidBook.items():
            if quantity > 10000 and posSize < HFT_POSITION_LIMIT:
                self.placeOrder(ob, "buy", float(price)+0.01, random.randint(1, HFT_BASE_ORDER_SIZE), "limit")
            
        for price, quantity in askBook.items():
            if quantity > 10000 and posSize > -HFT_POSITION_LIMIT:
                self.placeOrder(ob, "sell", float(price)-0.01, random.randint(1, HFT_BASE_ORDER_SIZE), "limit")
        
        # trade asymmetry in order books
        bidSize = ob.get_bidSize()
        askSize = ob.get_askSize()
        if bidSize + askSize > 0:
            imbalance = (bidSize - askSize)/(bidSize + askSize)
        else:
            imbalance = 0

        if imbalance > 0.6 and posSize < HFT_POSITION_LIMIT:
            q =  random.randint(1, HFT_BASE_ORDER_SIZE)
            self.placeOrder(ob, "buy", 100, q, "market")
            self.placeOrder(ob, "sell", ob.get_best_ask()+0.05, q, "limit")
        elif imbalance < -0.6 and posSize > - HFT_POSITION_LIMIT:
            q =  random.randint(1, HFT_BASE_ORDER_SIZE)
            self.placeOrder(ob, "sell", 100, q, "market")
            self.placeOrder(ob, "sell", ob.get_best_bid()-0.05, q, "limit")

    def tradeTheNews(self, market, sentiment_score):
        #  If sentiment is above 0.7 or below 0.3, make the HFT front run the trade by market buying/selling 
        if sentiment_score <= 0.3:
            quantity = HFT_BASE_ORDER_SIZE * (0.5-max(0,sentiment_score))*10
            self.placeOrder(markets[market], "sell", 100, int(quantity), "market")

        elif sentiment_score >= 0.7:
            quantity = HFT_BASE_ORDER_SIZE * (1-min(1,sentiment_score))*10

            self.placeOrder(markets[market], "buy", 100, int(quantity), "market")