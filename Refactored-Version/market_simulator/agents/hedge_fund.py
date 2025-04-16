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
        # Trades so each asset trades near its fair value.
        fv_gaps = {}
        fvs = {}
        for asset in assets:
            fv = calculate_fair_value(asset)

            if fv is not None and asset != "SPY":
                fvs[asset] = fv
                fv_gaps[asset] = (markets[asset].lastPrice - fv)/fv
            
        # Find assets with highest and lowest YTM
        mostOverpriced = max(fv_gaps, key=fv_gaps.get)
        mostUnderpriced = min(fv_gaps, key=fv_gaps.get)
        
        # If more than 3% away from fair value, should do the mean reversion trade
        if fv_gaps[mostOverpriced] < 0.03 or fv_gaps[mostUnderpriced] > -0.03:
            self.removePositionTargets()
        if fv_gaps[mostOverpriced] > 0.03:
            print("SELLING, ", mostOverpriced, " because: ", fv_gaps[mostOverpriced])
            self.targetPosition(markets[mostOverpriced], "sell", fvs[mostOverpriced], HF_POSITION_LIMIT)
        if fv_gaps[mostUnderpriced] < -0.03:
            print("BUYING, ", mostUnderpriced, " because: ", fv_gaps[mostUnderpriced])
            self.targetPosition(markets[mostUnderpriced], "buy", fvs[mostUnderpriced], HF_POSITION_LIMIT)