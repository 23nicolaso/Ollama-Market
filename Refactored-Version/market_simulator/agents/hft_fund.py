import random
from market_simulator.agents.executional_trader import ExecutionalTrader
from market_simulator.utils.market_utils import markets
from market_simulator.config import SPREADS, HFT_BASE_ORDER_SIZE

class HFTFund(ExecutionalTrader):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)
        self.intendedOrders = {}
    
    def estimateImportance(self, rt):
        return max(0, min(10, rt.estimateImportance()))
    
    def estimateSentiment(self, rt, market):
        return max(0, min(1, rt.estimateSentiment(markets[market])))

    def tradeTheNews(self, market, rt):
        estimated_price_impact = self.estimateImportance(rt)*(SPREADS[markets[market].asset])*(abs(self.estimateSentiment(rt, market)-0.5))*10
        #  If sentiment is above 0.9 or below 0.1, make the HFT front run the trade by market buying/selling and then doing a smart execution to close
        if self.estimateSentiment(rt, market) <= 0.3:
            current_price = markets[market].getLastPrice()
            quantity = self.estimateImportance(rt)*HFT_BASE_ORDER_SIZE
            if markets[market].asset == "SPY":
                print("selling " + str(quantity) + " shares in " + markets[market].asset + " at " + str(current_price-estimated_price_impact))

            self.placeOrder(markets[market], "sell", current_price, int(quantity), "market")
            self.executeTradeInLegs(markets[market], "sell", current_price, int(quantity))

        elif self.estimateSentiment(rt, market) >= 0.7:
            current_price = markets[market].getLastPrice()

            quantity = self.estimateImportance(rt)*HFT_BASE_ORDER_SIZE
            self.placeOrder(markets[market], "buy", current_price, int(quantity), "market")
            self.executeTradeInLegs(markets[market], "buy", current_price, int(quantity))