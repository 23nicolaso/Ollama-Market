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
        current_price = markets[market].getLastPrice()
        
        # Check current position against position limit
        current_position = self.account.getPosition(market)
        position_limit = TA_POSITION_LIMIT
        
        if current_price > mean_price + std_dev*2:
            order_size = ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE)
            # Check if selling would exceed position limit on the short side
            if current_position - order_size >= -position_limit:
                self.placeOrder(markets[market], "sell", mean_price + std_dev*2, order_size, "market")
        elif current_price < mean_price - std_dev*2:
            order_size = ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE)
            # Check if buying would exceed position limit on the long side
            if current_position + order_size <= position_limit:
                self.placeOrder(markets[market], "buy", mean_price - std_dev*2, order_size, "market")

        # # if price changed by over 0.5 in 100 ticks, mean revert by fading the trade
        # if len(price_list) >= 100:
        #     price_change = price_list[-1] - price_list[-100]
        #     order_size = ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE)
        #     if price_change > 0.5:
        #         direction = "sell" if self.account.getPosition(market) > 0 else "buy"
        #         # Check position limits before placing order
        #         if (direction == "sell" and current_position - order_size >= -position_limit) or \
        #            (direction == "buy" and current_position + order_size <= position_limit):
        #             self.placeOrder(markets[market], direction, current_price, order_size, "market")
        #     elif price_change < -0.5:
        #         direction = "buy" if self.account.getPosition(market) > 0 else "sell"
        #         # Check position limits before placing order
        #         if (direction == "sell" and current_position - order_size >= -position_limit) or \
        #            (direction == "buy" and current_position + order_size <= position_limit):
        #             self.placeOrder(markets[market], direction, current_price, order_size, "market")

        # # trade on cross of close and moving average of 500 ticks
        # if len(price_list) >= 500:
        #     stma = sum(price_list[-500:]) / 500
        #     order_size = ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE)
            
        #     # If price crosses above MA, buy
        #     if current_price > stma and price_list[-2] <= stma:
        #         # Check if buying would exceed position limit
        #         if current_position + order_size <= position_limit:
        #             self.placeOrder(markets[market], "buy", current_price, order_size, "market")
                
        #     # If price crosses below MA, sell    
        #     elif current_price < stma and price_list[-2] >= stma:
        #         # Check if selling would exceed position limit
        #         if current_position - order_size >= -position_limit:
        #             self.placeOrder(markets[market], "sell", current_price, order_size, "market")

        # # If price is near a key level (nearest number), trade off the level
        # if abs(round(current_price) - current_price) < 0.1:
        #     stma = sum(price_list[-200:]) / 200
        #     order_size = ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE)
            
        #     if stma > round(current_price):
        #         # Check if selling would exceed position limit
        #         if current_position - order_size >= -position_limit:
        #             self.placeOrder(markets[market], "sell", round(current_price), order_size, "market")
        #     elif stma < round(current_price):
        #         # Check if buying would exceed position limit
        #         if current_position + order_size <= position_limit:
        #             self.placeOrder(markets[market], "buy", round(current_price), order_size, "market")

        # # If 2% drop over past 500 ticks, execute a dip-buying strategy
        # if len(price_history[market]) >= 500:
        #     start_price = price_history[market][-500]
        #     current_price = price_history[market][-1]
        #     price_drop = (start_price - current_price) / start_price
        #     price_spike = (current_price - start_price) / start_price
            
        #     if price_drop > 0.02:
        #         order_size = int(TA_MEGA_ORDER_SIZE*random.uniform(0.5, 1.5))
        #         # Check if buying would exceed position limit
        #         if current_position + order_size <= position_limit:
        #             self.executeTradeInLegs(markets[market], "buy", current_price, order_size)

        #     if price_spike > 0.02:
        #         order_size = int(TA_MEGA_ORDER_SIZE*random.uniform(0.5, 1.5))
        #         # Check if selling would exceed position limit
        #         if current_position - order_size >= -position_limit:
        #             self.executeTradeInLegs(markets[market], "sell", current_price, order_size)
                
        # Limit the number of limitorders placed to save on compute
        max_orders = 200
        if len(markets[market].bids) + len(markets[market].asks) > max_orders:
            # Cancel all TA orders
            markets[market].cancelOrdersByAccount(self.account.accountID)
            return
