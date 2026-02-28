import random
from market_simulator.agents.executional_trader import ExecutionalTrader
from market_simulator.utils.market_utils import markets, assets
from market_simulator.config import HF_POSITION_LIMIT, INITIAL_PRICES, SPY_INCLUDED_ASSETS, HF_BASE_ORDER_SIZE
class HedgeFund(ExecutionalTrader):
    def __init__(self, accountID, cash, strategy_type):
        super().__init__(accountID, cash)
        self.strategy_type = strategy_type
        self.position_limits = {asset: HF_POSITION_LIMIT for asset in assets}
        self.last_market_return_profile = None
    
    def set_market_return_profile(self, return_profile):
        self.last_market_return_profile = return_profile
    
    def calculate_target_positions(self):
        if self.strategy_type == "mean_reversion":
            self.mean_reversion_strategy()

    def place_bets(self, predictions):
        for asset in SPY_INCLUDED_ASSETS:
            if predictions[asset] > 0:
                self.targetPosition(markets[asset], "buy", markets[asset].last_price+1, markets[asset].last_price, HF_BASE_ORDER_SIZE, True)
            else:
                self.targetPosition(markets[asset], "sell", markets[asset].last_price-1, markets[asset].last_price, HF_BASE_ORDER_SIZE, True)

    def close_bets(self):
        for asset in SPY_INCLUDED_ASSETS:
            self.targetPosition(markets[asset], "buy", markets[asset].last_price, markets[asset].last_price, 0, True)

    def mean_reversion_strategy(self):
        if self.last_market_return_profile:
            for asset in SPY_INCLUDED_ASSETS:
                pct_change = (markets[asset].last_price-INITIAL_PRICES[asset])/INITIAL_PRICES[asset]
                expected_change = self.last_market_return_profile[asset]
                delta = pct_change - expected_change
                if delta > 0.01:
                    self.placeOrder(markets[asset], "sell", markets[asset].last_price-0.1, HF_POSITION_LIMIT, "limit")
                elif delta < -0.01:
                    self.placeOrder(markets[asset], "buy", markets[asset].last_price+0.1, HF_POSITION_LIMIT, "limit")
                elif abs(delta) < 0.005:
                    self.targetPosition(markets[asset], "buy", round(INITIAL_PRICES[asset]*(1+expected_change),2), markets[asset].last_price, 0)