import random
from market_simulator.agents.executional_trader import ExecutionalTrader
from market_simulator.utils.market_utils import price_history, markets, assets
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
                self.targetPosition(markets[asset], "buy", markets[asset].last_price+0.5, HF_BASE_ORDER_SIZE)
            else:
                self.targetPosition(markets[asset], "sell", markets[asset].last_price-0.5, HF_BASE_ORDER_SIZE)

    def close_bets(self):
        for asset in SPY_INCLUDED_ASSETS:
            self.targetPosition(markets[asset], "buy", markets[asset].last_price, 0)

    def mean_reversion_strategy(self):
        if self.last_market_return_profile:
            for asset in SPY_INCLUDED_ASSETS:
                pct_change = (markets[asset].last_price-INITIAL_PRICES[asset])/INITIAL_PRICES[asset]
                expected_change = self.last_market_return_profile[asset]
                delta = pct_change - expected_change
                if delta > 0.02:
                    self.targetPosition(markets[asset], "sell", markets[asset].last_price, HF_POSITION_LIMIT)
                elif delta < -0.02:
                    self.targetPosition(markets[asset], "buy", markets[asset].last_price, HF_POSITION_LIMIT)
                