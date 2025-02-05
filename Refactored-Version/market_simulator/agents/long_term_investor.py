from market_simulator.agents.executional_trader import ExecutionalTrader
from market_simulator.utils.market_utils import assets, markets
import random
from market_simulator.config import LT_INVESTOR_MAX_ORDER_SIZE, NUM_SHARES

class LongTermInvestor(ExecutionalTrader):
    def __init__(self, account_id="LongTermInvestor", cash=10000000):
        super().__init__(account_id, cash)
        # Start with 10% of all assets
        for asset in assets:
            self.account.positions[asset] = int(0.1 * NUM_SHARES[asset]) # Start with 10% of 1M shares for each asset

    def trade(self, market):
        """Trade based on long-term investment strategy with panic selling"""

        if random.random() < 0.025:
            # Buy random amount on random intervals
            quantity = random.randint(1, LT_INVESTOR_MAX_ORDER_SIZE)
            self.placeOrder(market, "buy", market.getLastPrice(), quantity, "market")
    
    def tradeNews(self, ticker, sentiment_score, importance):
        if sentiment_score <= 0.2 and importance > 8:
            position = self.account.getPosition(ticker)
            proportion = 0.005*max(1, importance)
            self.executeTradeInLegs(markets[ticker], "sell", markets[ticker].getLastPrice(), int(proportion*position))