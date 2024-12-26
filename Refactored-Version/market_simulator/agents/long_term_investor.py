from market_simulator.agents.executional_trader import ExecutionalTrader
from market_simulator.utils.market_utils import assets
import random
from market_simulator.config import LT_INVESTOR_MAX_ORDER_SIZE, SPREADS

class LongTermInvestor(ExecutionalTrader):
    def __init__(self, account_id="LongTermInvestor", cash=10000000):
        super().__init__(account_id, cash)

    def trade(self, market, retail_trader):
        """Trade based on long-term investment strategy with panic selling"""
        # Check if sentiment is very low (panic threshold)
        if retail_trader.retailSentimentScore[market.asset] <= 0.3:
            # Panic sell everything for this asset
            position = self.account.getPosition(market.asset)
            if position > 0:
                self.placeOrder(market, "sell", market.getLastPrice(), round(position*0.1), "market")
            return

        else:
            if random.random() < 0.05:
                # Buy random amount on random intervals
                quantity = random.randint(1, LT_INVESTOR_MAX_ORDER_SIZE)
                self.placeOrder(market, "buy", market.getLastPrice(), quantity, "market")