from market_simulator.agents.executional_trader import ExecutionalTrader
from market_simulator.utils.market_utils import assets, markets
import random
from market_simulator.config import LT_INVESTOR_MAX_ORDER_SIZE, LTI_POS_LIMIT, INITIAL_PRICES

class LongTermInvestor(ExecutionalTrader):
    def __init__(self, account_id="LongTermInvestor", cash=10000000):
        super().__init__(account_id, cash)
        # Start with 60% of capital spread across all assets (dollar-budget, cash deducted)
        budget_per_asset = int(cash * 0.6 / len(assets))
        for asset in assets:
            shares = int(budget_per_asset / INITIAL_PRICES[asset])
            if shares > 0:
                cost = shares * INITIAL_PRICES[asset]
                self.account.addPosition(asset, shares)
                self.account.addPosition("CASH", -cost)

    def trade(self, market):
        """In reality, they might have some kind of strategy, but here its really just random"""
        quantity = random.randint(100, 100000)

        if random.random() < 0.1:
            # Buy random amount on random intervals
            self.placeOrder(market, random.choice(["buy", "sell"]), market.last_price, quantity, "iceberg", 100)
    
    def tradeNews(self, ticker, sentiment_score, importance):
        return