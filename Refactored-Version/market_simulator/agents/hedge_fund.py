import random
from market_simulator.agents.executional_trader import ExecutionalTrader
from market_simulator.utils.market_utils import price_history, markets, assets, calculate_r_adj_ytm
from market_simulator.config import HF_POSITION_LIMIT, SPREADS

class HedgeFund(ExecutionalTrader):
    def __init__(self, accountID, cash, strategy_type):
        super().__init__(accountID, cash)
        self.strategy_type = strategy_type
        self.position_limits = {asset: HF_POSITION_LIMIT for asset in assets}
        
    def update_positions(self, market):
        return
    
    def calculate_target_positions(self):
        if self.strategy_type == "mean_reversion":
            self.mean_reversion_strategy()
        elif self.strategy_type == "macro":
            self.macro_strategy()

    def mean_reversion_strategy(self):
        # Calculate YTM for all assets
        ytms = {}
        for asset in assets:
            result = calculate_r_adj_ytm(asset)
            if result is not None:
                ytms[asset] = result
            
        # Find assets with highest and lowest YTM
        highest_ytm_asset = max(ytms, key=ytms.get)
        lowest_ytm_asset = min(ytms, key=ytms.get)
        
        # Short the highest YTM asset
        self.placeOrder(markets[highest_ytm_asset], "buy", 0.01, 100, "market")
        self.placeOrder(markets[lowest_ytm_asset], "sell", 0.01, 100, "market")
        # print(ytms)
        # print("BUYING: ", highest_ytm_asset, ". SELLING: ", lowest_ytm_asset)