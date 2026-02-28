from math import ceil
import random
from market_simulator.agents.executional_trader import ExecutionalTrader
from market_simulator.utils.market_utils import price_history, markets
from market_simulator.config import TA_LARGE_ORDER_SIZE, TA_MEGA_ORDER_SIZE, TA_POSITION_LIMIT

class TATrader(ExecutionalTrader):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)
        self.intendedOrders = {}
        self.conditionalOrders = {}

    def manageTATrades(self, market):
        # Calculate mean and standard deviation of price history
        mean_price = price_history[market.asset].mean()
        std_dev = min(price_history[market.asset].std(), 0.05)
        current_price = markets[market.asset].last_price
        if not isinstance(current_price, float):
            current_price = 0

        # Check current position against position limit
        current_position = self.account.getPosition(market.asset)
        position_limit = TA_POSITION_LIMIT
        
        # Mean reversion trade off VWAP levels
        order_size = TA_LARGE_ORDER_SIZE
        # Check if selling would exceed position limit on the short side
        self.cancelAllOrders(markets[market.asset])
        if current_position - order_size >= -position_limit:
            self.placeOrder(markets[market.asset], "sell", mean_price + std_dev*2, 1, "limit")
        if current_position + order_size <= position_limit:
            self.placeOrder(markets[market.asset], "buy", mean_price - std_dev*2, 1, "limit")