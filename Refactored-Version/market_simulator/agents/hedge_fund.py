import random
from market_simulator.agents.executional_trader import ExecutionalTrader
from market_simulator.utils.market_utils import price_history, markets, assets, calculate_r_adj_ytm, calculate_fair_value
from market_simulator.config import HF_POSITION_LIMIT, SPREADS, RFR, HF_BASE_ORDER_SIZE

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
        # Trades so each asset returns roughly the same as the risk free asset, after adjusting for inflation and risk. 
        ytms = {}
        fvs = {}
        for asset in assets:
            result = calculate_r_adj_ytm(asset)
            fv = calculate_fair_value(asset)
            if result is not None:
                ytms[asset] = result
            if fv is not None:
                fvs[asset] = fv
            
        # Find assets with highest and lowest YTM
        highest_ytm_asset = max(ytms, key=ytms.get)
        lowest_ytm_asset = min(ytms, key=ytms.get)
        
        # If more than 1% away from RFR, should do the mean reversion trade
        if ytms[lowest_ytm_asset] < 1 + RFR - 0.01:
            # print("SELLING, ", lowest_ytm_asset, " because: ", ytms[lowest_ytm_asset])
            self.executeTradeInLegs(markets[lowest_ytm_asset], "sell", fvs[asset], HF_BASE_ORDER_SIZE)
        if ytms[highest_ytm_asset] > 1 + RFR + 0.01:
            # print("BUYING, ", highest_ytm_asset, " because: ", ytms[highest_ytm_asset])
            self.executeTradeInLegs(markets[highest_ytm_asset], "buy", fvs[asset], HF_BASE_ORDER_SIZE)

        # print(ytms)