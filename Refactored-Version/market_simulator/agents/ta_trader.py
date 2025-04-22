from math import ceil
import random
from market_simulator.agents.executional_trader import ExecutionalTrader
from market_simulator.utils.market_utils import price_history, markets
from market_simulator.config import TA_POSITION_LIMIT, TA_LARGE_ORDER_SIZE, TA_MEGA_ORDER_SIZE

class TATrader(ExecutionalTrader):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)
        self.intendedOrders = {}
        self.conditionalOrders = {}

    def manageTATrades(self, market):
        # Calculate mean and standard deviation of price history
        mean_price = price_history[market].mean()
        std_dev = price_history[market].std()
        current_price = markets[market].last_price
        if not isinstance(current_price, float):
            current_price = 0
        st_mean = price_history[market].mean(n=50)

        # Check current position against position limit
        current_position = self.account.getPosition(market)
        position_limit = TA_POSITION_LIMIT
        
        # Mean reversion trade off VWAP levels
        if current_price > mean_price + std_dev*2:
            order_size = ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE)
            # Check if selling would exceed position limit on the short side
            if current_position - order_size >= -position_limit:
                self.executeTradeInLegs(markets[market], "sell", mean_price + std_dev*2, order_size)
        elif current_price < mean_price - std_dev*2:
            order_size = ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE)
            # Check if buying would exceed position limit on the long side
            if current_position + order_size <= position_limit:
                self.executeTradeInLegs(markets[market], "buy", mean_price - std_dev*2, order_size)
        else:
            self.flattenIntendedExecutions()

        # If price is near a key level (nearest number), trade off the level
        if abs(round(current_price) - current_price) < 0.1:
            order_size = ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE)
            
            if st_mean > round(current_price):
                # Check if selling would exceed position limit
                if current_position - order_size >= -position_limit:
                    self.placeOrder(markets[market], "sell", round(current_price), order_size, "market")
            elif st_mean < round(current_price):
                # Check if buying would exceed position limit
                if current_position + order_size <= position_limit:
                    self.placeOrder(markets[market], "buy", round(current_price), order_size, "market")

        # Trade the MA cross
        crossover = price_history[market].crossed_over_mean(n=50)
        if crossover:
            order_size = ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE)
            if current_price > st_mean:
                # buy
                if current_position + order_size <= position_limit:
                    self.placeOrder(markets[market], "buy", round(current_price), order_size, "market")
            else:
                # sell
                if current_position - order_size >= -position_limit:
                    self.placeOrder(markets[market], "sell", round(current_price), order_size, "market")
            