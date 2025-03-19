import random
from market_simulator.agents.executional_trader import ExecutionalTrader
from market_simulator.utils.market_utils import price_history, markets, assets
from market_simulator.config import HF_POSITION_LIMIT, SPREADS

class HedgeFund(ExecutionalTrader):
    def __init__(self, accountID, cash, strategy_type):
        super().__init__(accountID, cash)
        self.strategy_type = strategy_type
        self.position_limits = {asset: HF_POSITION_LIMIT for asset in assets}
        self.target_positions = {asset: 0 for asset in assets}
        
    def update_positions(self, market):
        current_position = self.account.getPosition(market)
        target_position = self.target_positions[market]
        
        if current_position < target_position:
            if current_position - target_position < 10000:
                quantity = max(1, target_position - current_position)
                self.placeOrder(markets[market], "buy", markets[market].lastPrice, quantity, "market")
            else:
                quantity = max(1, target_position - current_position)
                self.executeTradeInLegs(markets[market], "buy", markets[market].lastPrice, quantity)
        elif current_position > target_position:
            # Need to sell
            if current_position - target_position < 10000:
                quantity = max(1, current_position - target_position)
                self.placeOrder(markets[market], "sell", markets[market].lastPrice, quantity, "market")
            else:
                quantity = max(1, current_position - target_position)
                self.executeTradeInLegs(markets[market], "sell", markets[market].lastPrice, quantity)
        
        self.updateOrdersInLegs(markets[market])
        self.partialExecuteMarket(markets[market])

    def calculate_target_positions(self):
        if self.strategy_type == "mean_reversion":
            self.mean_reversion_strategy()
        elif self.strategy_type == "macro":
            self.macro_strategy()

    def mean_reversion_strategy(self):
        return
        # for asset in assets:
        #     if len(price_history[asset]) < 50:
        #         continue
            
        #     # Calculate mean and standard deviation
        #     mean_price = sum(price_history[asset][-50:]) / 50
        #     current_price = markets[asset].lastPrice
            
        #     # If price is significantly above mean, sell; if below, buy
        #     deviation = (current_price - mean_price) / mean_price
        #     self.target_positions[asset] = -int(deviation * self.position_limits[asset])