from market_simulator.agents.executional_trader import ExecutionalTrader
from market_simulator.utils.market_utils import assets, markets
import random
from market_simulator.config import LT_INVESTOR_MAX_ORDER_SIZE, NUM_SHARES, LTI_POS_LIMIT

class LongTermInvestor(ExecutionalTrader):
    def __init__(self, account_id="LongTermInvestor", cash=10000000):
        super().__init__(account_id, cash)
        # Start with 10% of all assets
        for asset in assets:
            self.account.positions[asset] = int(0.1 * NUM_SHARES[asset]) # Start with 10% of 1M shares for each asset

    def trade(self, market):
        """In reality, they might have some kind of strategy, but here its really just random"""
        quantity = random.randint(1, LT_INVESTOR_MAX_ORDER_SIZE)

        if self.account.positions[market.asset] + quantity < LTI_POS_LIMIT:
            if random.random() < 0.001:
                # Buy random amount on random intervals
                self.placeOrder(market, random.choice(["buy", "sell"]), market.lastPrice, quantity, "limit")
    
    def tradeNews(self, ticker, sentiment_score, importance):
        if sentiment_score <= 0.1 and importance == 10: 
            self.executeTradeInLegs(ticker, "sell", 0, self.account.getPosition(ticker)*0.05) # panic sell lol