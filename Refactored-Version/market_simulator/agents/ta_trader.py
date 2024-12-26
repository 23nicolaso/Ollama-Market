from math import ceil
import random
from market_simulator.agents.executional_trader import ExecutionalTrader
from market_simulator.utils.market_utils import price_history, spreads_by_market, markets
from market_simulator.config import TA_POSITION_LIMIT, TA_SMALL_ORDER_SIZE, TA_LARGE_ORDER_SIZE

class TATrader(ExecutionalTrader):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)
        self.intendedOrders = {}
        self.conditionalOrders = {}

    def manageTATrades(self, market):
        # Calculate mean and standard deviation of price history
        price_list = price_history[market][-20:]
        mean_price = sum(price_list) / len(price_list)
        std_dev = (sum((x - mean_price) ** 2 for x in price_list) / len(price_list)) ** 0.5
        current_price = markets[market].getLastPrice()

        if len(price_history[market]) >= 30:
            price_30_ticks_ago = price_history[market][-30]
            price_change = abs(current_price - price_30_ticks_ago)

            if price_change > 0.5:
                # Add limit orders to calm down the move/push price back to average
                if current_price > price_30_ticks_ago:
                    # Price increased, add sell market orders
                    self.placeOrder(markets[market], "sell", current_price-0.01, ceil(random.uniform(0.5, 1.5)*TA_SMALL_ORDER_SIZE), "limit")
                else:
                    # Price decreased, add buy market orders
                    self.placeOrder(markets[market], "buy", current_price+0.01, ceil(random.uniform(0.5, 1.5)*TA_SMALL_ORDER_SIZE), "limit")

        # Calculate 8-period moving average
        if len(price_list) >= 30:
            ma_8 = sum(price_list[-8:]) / 8
            
            # Determine action based on current price relative to 8-period MA
            if current_price < ma_8:
                if ma_8 - current_price <  spreads_by_market[market]*5:
                    # Buy market
                    self.placeOrder(markets[market], "buy", current_price, ceil(random.uniform(0.5, 1.5)*TA_SMALL_ORDER_SIZE), "market")
                    self.placeOrder(markets[market], "sell", current_price+random.uniform(0.05, 0.1), ceil(random.uniform(0.5, 1.5)*TA_SMALL_ORDER_SIZE), "limit")
            else:
                if current_price - ma_8 < spreads_by_market[market]*5:
                    # Sell market
                    self.placeOrder(markets[market], "sell", current_price, ceil(random.uniform(0.5, 1.5)*TA_SMALL_ORDER_SIZE), "market")
                    self.placeOrder(markets[market], "buy", current_price-random.uniform(0.05, 0.1), ceil(random.uniform(0.5, 1.5)*TA_SMALL_ORDER_SIZE), "limit")
        
        if len(price_list) > 30:
            ma_30 = sum(price_list[-30:]) / 30
            if current_price < ma_30:
                # Sell market
                self.placeOrder(markets[market], "sell", current_price, ceil(random.uniform(0.5, 1.5)*TA_SMALL_ORDER_SIZE), "limit")
            else:
                # Buy market
                self.placeOrder(markets[market], "buy", current_price, ceil(random.uniform(0.5, 1.5)*TA_SMALL_ORDER_SIZE), "limit")

        if current_price > mean_price + std_dev*2 and self.account.getPosition(market) > -TA_POSITION_LIMIT:
            self.placeOrder(markets[market], "sell", mean_price + std_dev*2, ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE), "market")
            self.placeOrder(markets[market], "buy", mean_price + std_dev*2-random.uniform(0.05, 0.1), ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE), "limit")
        elif current_price < mean_price - std_dev*2 and self.account.getPosition(market) < TA_POSITION_LIMIT:
            self.placeOrder(markets[market], "buy", mean_price - std_dev*2, ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE), "market")
            self.placeOrder(markets[market], "sell", mean_price - std_dev*2+random.uniform(0.05, 0.1), ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE), "limit")

        vwap = sum(price_history[market]) / len(price_history[market])
        stdev = (sum((x - vwap) ** 2 for x in price_history[market]) / len(price_history[market])) ** 0.5
        stma = sum(price_list[-10:]) / 10
        vndo = (stma - vwap) / max(stdev,0.001)

        if current_price > vwap + (vndo+1)*stdev and self.account.getPosition(market) > -TA_POSITION_LIMIT:
            self.placeOrder(markets[market], "sell", vwap + stdev*2, ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE), "limit")
        elif current_price < vwap + (vndo-1)*stdev and self.account.getPosition(market) < TA_POSITION_LIMIT:
            self.placeOrder(markets[market], "buy", vwap - stdev*2, ceil(random.uniform(0.5, 1.5)*TA_LARGE_ORDER_SIZE), "limit")

        # Limit the number of limitorders placed to save on compute
        max_orders = 200
        if len(markets[market].bids) + len(markets[market].asks) > max_orders:
            # Cancel all TA orders
            for bid_level in markets[market].bids.values():
                bid_level.cancelOrdersFromID("TA Trading Firm")
            for ask_level in markets[market].asks.values():
                ask_level.cancelOrdersFromID("TA Trading Firm")
            return
