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
        ob = markets[ticker]

        # trade asymmetry in order books
        bidSize = ob.bid_size
        askSize = ob.ask_size
        if bidSize + askSize > 0:
            imbalance = (bidSize - askSize)/(bidSize + askSize)
        else:
            imbalance = 0

        if imbalance > 0.6:
            self.targetPosition(markets[ticker], "buy", markets[ticker].last_price+0.5, HFT_POSITION_LIMIT)
        elif imbalance < -0.6:
            self.targetPosition(markets[ticker], "sell", markets[ticker].last_price-0.5, HFT_POSITION_LIMIT)

    def tradeTheNews(self, market, sentiment_score):
        #  If sentiment is above 0.7 or below 0.3, make the HFT front run the trade by market buying/selling 
        if sentiment_score <= 0.3:
            quantity = HFT_BASE_ORDER_SIZE * (0.5-max(0,sentiment_score))*10
            self.placeOrder(markets[market], "sell", 1, int(quantity), "market")

        elif sentiment_score >= 0.7:
            quantity = HFT_BASE_ORDER_SIZE * (1-min(1,sentiment_score))*10
            self.placeOrder(markets[market], "buy", 1, int(quantity), "market")