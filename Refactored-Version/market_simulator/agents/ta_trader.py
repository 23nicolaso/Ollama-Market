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

        if current_price > mean_price + std_dev*2 and self.account.getPosition(market) > -TA_POSITION_LIMIT:
            self.placeOrder(markets[market], "sell", mean_price + std_dev*2, ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE), "limit")
        elif current_price < mean_price - std_dev*2 and self.account.getPosition(market) < TA_POSITION_LIMIT:
            self.placeOrder(markets[market], "buy", mean_price - std_dev*2, ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE), "limit")

        # trade on cross of close and moving average of 1000 ticks
        if len(price_list) >= 1000:
            stma = sum(price_list[-1000:]) / 1000
            
            # If price crosses above MA, buy
            if current_price > stma and price_list[-2] <= stma and self.account.getPosition(market) < TA_POSITION_LIMIT:
                self.placeOrder(markets[market], "buy", current_price, ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE), "market")
                
            # If price crosses below MA, sell    
            elif current_price < stma and price_list[-2] >= stma and self.account.getPosition(market) > -TA_POSITION_LIMIT:
                self.placeOrder(markets[market], "sell", current_price, ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE), "market")

        # If 2% drop over past 1000 ticks, execute a dip-buying strategy
        if len(price_history[market]) >= 1000:
            start_price = price_history[market][-1000]
            current_price = price_history[market][-1]
            price_drop = (start_price - current_price) / start_price
            
            if price_drop > 0.02:
                self.executeTradeInLegs(markets[market], "buy", current_price, int(TA_MEGA_ORDER_SIZE*random.uniform(0.5, 1.5)))

        # Limit the number of limitorders placed to save on compute
        max_orders = 200
        if len(markets[market].bids) + len(markets[market].asks) > max_orders:
            # Cancel all TA orders
            for bid_level in markets[market].bids.values():
                bid_level.cancelOrdersFromID("TA Trading Firm")
            for ask_level in markets[market].asks.values():
                ask_level.cancelOrdersFromID("TA Trading Firm")
            return
