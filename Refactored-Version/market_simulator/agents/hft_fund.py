import random
from market_simulator.agents.executional_trader import ExecutionalTrader
from market_simulator.utils.market_utils import markets
from market_simulator.config import SPREADS, HFT_BASE_ORDER_SIZE

class HFTFund(ExecutionalTrader):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)
        self.intendedOrders = {}

    def tradeTheNews(self, market, sentiment_score):
        estimated_price_impact = 5*(SPREADS[markets[market].asset])*(abs(sentiment_score-0.5))*10
        #  If sentiment is above 0.7 or below 0.3, make the HFT front run the trade by market buying/selling 
        if sentiment_score <= 0.3:
            current_price = markets[market].getLastPrice()
            quantity = 5*HFT_BASE_ORDER_SIZE
            if markets[market].asset == "SPY":
                print("selling " + str(quantity) + " shares in " + markets[market].asset + " at " + str(current_price-estimated_price_impact))

            self.placeOrder(markets[market], "sell", current_price, int(quantity), "market")
            self.executeTradeInLegs(markets[market], "buy", current_price, int(quantity))

        elif sentiment_score >= 0.7:
            current_price = markets[market].getLastPrice()

            quantity = 5*HFT_BASE_ORDER_SIZE
            self.placeOrder(markets[market], "buy", current_price, int(quantity), "market")
            self.executeTradeInLegs(markets[market], "sell", current_price, int(quantity))