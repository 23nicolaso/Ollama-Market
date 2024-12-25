import random
from market_simulator.agents.executional_trader import ExecutionalTrader
from market_simulator.utils.market_utils import price_history, markets, economic_health_by_market, assets

class HedgeFund(ExecutionalTrader):
    def __init__(self, accountID, cash, strategy_type):
        super().__init__(accountID, cash)
        self.strategy_type = strategy_type
        self.position_limits = {
            "Simula 500": 100000,
            "Rivala ETF": 100000,
            "Allia ETF": 100000,
            "Factoria ETF": 100000,
            "Gold": 100000
        }
        self.target_positions = {asset: 0 for asset in assets}
        
    def update_positions(self, market):
        current_position = self.account.getPosition(market)
        target_position = self.target_positions[market]
        
        if current_position < target_position:
            # Need to buy
            quantity = min(10000, target_position - current_position)
            self.executeTradeInLegs(markets[market], "buy", markets[market].getLastPrice(), quantity)
        elif current_position > target_position:
            # Need to sell
            quantity = min(10000, current_position - target_position)
            self.executeTradeInLegs(markets[market], "sell", markets[market].getLastPrice(), quantity)
        
        self.updateOrdersInLegs(markets[market])
        self.partialExecuteMarket(markets[market])

    def calculate_target_positions(self):
        if self.strategy_type == "mean_reversion":
            self.mean_reversion_strategy()
        elif self.strategy_type == "macro":
            self.macro_strategy()

    def mean_reversion_strategy(self):
        for asset in assets:
            if len(price_history[asset]) < 50:
                continue
            
            # Calculate mean and standard deviation
            mean_price = sum(price_history[asset][-50:]) / 50
            current_price = markets[asset].getLastPrice()
            
            # If price is significantly above mean, sell; if below, buy
            deviation = (current_price - mean_price) / mean_price
            self.target_positions[asset] = -int(deviation * self.position_limits[asset])

    def macro_strategy(self):
        for asset in assets:
            # Use economic health as a indicator
            economic_health = economic_health_by_market[asset]
            
            # Use economic health for position sizing
            score = (1 - economic_health) * 0.5  
            self.target_positions[asset] = int((score - 0.5) * 2 * self.position_limits[asset])
