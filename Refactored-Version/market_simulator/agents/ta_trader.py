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
        price_list = price_history[market][-100:]
        mean_price = sum(price_list) / len(price_list)
        std_dev = (sum((x - mean_price) ** 2 for x in price_list) / len(price_list)) ** 0.5
        current_price = markets[market].getLastPrice()

        if current_price > mean_price + std_dev*2 :
            self.placeOrder(markets[market], "sell", mean_price + std_dev*2, ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE), "market")
        elif current_price < mean_price - std_dev*2:
            self.placeOrder(markets[market], "buy", mean_price - std_dev*2, ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE), "market")

        # if price changed by over 0.5 in 100 ticks, mean revert by fading the trade
        if len(price_list) >= 100:
            price_change = price_list[-1] - price_list[-100]
            if price_change > 0.5:
                self.placeOrder(markets[market], "sell" if self.account.getPosition(market) > 0 else "buy", current_price, ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE), "market")
            elif price_change < -0.5:
                self.placeOrder(markets[market], "buy" if self.account.getPosition(market) > 0 else "sell", current_price, ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE), "market")

        # trade on cross of close and moving average of 500 ticks
        if len(price_list) >= 500:
            stma = sum(price_list[-500:]) / 500
            
            # If price crosses above MA, buy
            if current_price > stma and price_list[-2] <= stma:
                self.placeOrder(markets[market], "buy", current_price, ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE), "market")
                
            # If price crosses below MA, sell    
            elif current_price < stma and price_list[-2] >= stma:
                self.placeOrder(markets[market], "sell", current_price, ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE), "market")

        # If price is near a key level (nearest number), trade off the level
        if abs(round(current_price) - current_price) < 0.1:

            stma = sum(price_list[-200:]) / 200
            if stma > round(current_price):
                self.placeOrder(markets[market], "sell", round(current_price), ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE), "market")
            elif stma < round(current_price):
                self.placeOrder(markets[market], "buy", round(current_price), ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE), "market")

        # If 2% drop over past 500 ticks, execute a dip-buying strategy
        if len(price_history[market]) >= 500:
            start_price = price_history[market][-500]
            current_price = price_history[market][-1]
            price_drop = (start_price - current_price) / start_price
            price_spike = (current_price - start_price) / start_price
            
            if price_drop > 0.02:
                self.executeTradeInLegs(markets[market], "buy", current_price, int(TA_MEGA_ORDER_SIZE*random.uniform(0.5, 1.5)))

            if price_spike > 0.02:
                self.executeTradeInLegs(markets[market], "sell", current_price, int(TA_MEGA_ORDER_SIZE*random.uniform(0.5, 1.5)))
                
        # Limit the number of limitorders placed to save on compute
        max_orders = 200
        if len(markets[market].bids) + len(markets[market].asks) > max_orders:
            # Cancel all TA orders
            markets[market].cancelOrdersByAccount(self.account.accountID)
            return
