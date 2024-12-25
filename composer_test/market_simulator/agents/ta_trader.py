import random
from market_simulator.agents.executional_trader import ExecutionalTrader
from market_simulator.utils.market_utils import price_history, spreads_by_market, markets

class TATrader(ExecutionalTrader):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)
        self.intendedOrders = {}
        self.conditionalOrders = {}

    def getConditionalOrdersInDirection(self, market, direction):
        if market not in self.conditionalOrders:
            return 0
        orders = [order for order in self.conditionalOrders[market] if order["direction"] == direction]
        return sum(order["quantity"] for order in orders)

    def placeConditionalOrder(self, orderBook, direction, price, quantity, priceCondition, conditionalDirection):
        if orderBook.asset not in self.conditionalOrders:
            self.conditionalOrders[orderBook.asset] = []
        self.conditionalOrders[orderBook.asset].append({"direction": direction, "price": price, "quantity": quantity, "priceCondition": priceCondition, "conditionalDirection": conditionalDirection})

    def manageTATrades(self, market):
        # Calculate mean and standard deviation of price history
        price_list = price_history[market][-20:]
        mean_price = sum(price_list) / len(price_list)
        std_dev = (sum((x - mean_price) ** 2 for x in price_list) / len(price_list)) ** 0.5
        quantity = random.randint(1, 10) * 10
        current_price = markets[market].getLastPrice()

        # Position conditional orders at the high of the past 500 ticks and low of the past 500 ticks to emulate stop loss / take profit orders getting hit
        if len(price_history[market]) > 300:
            high_lt = max(price_history[market][-300:])
            low_lt = min(price_history[market][-300:]) 
        
            if self.getConditionalOrdersInDirection(market, "sell") == 0 and self.account.getPosition(market) > 0:
                self.placeConditionalOrder(markets[market], "sell", low_lt, self.account.getPosition(market)//20, low_lt, "below")
            elif self.getConditionalOrdersInDirection(market, "buy") == 0 and self.account.getPosition(market) < 0:
                self.placeConditionalOrder(markets[market], "buy", high_lt, -self.account.getPosition(market)//20, high_lt, "above")
    
        if len(price_history[market]) >= 30:
            price_30_ticks_ago = price_history[market][-30]
            price_change = abs(current_price - price_30_ticks_ago) / max(1,price_30_ticks_ago)

            if price_change > 0.01:
                # Add limit orders to calm down the move/push price back to average
                if current_price > price_30_ticks_ago:
                    # Price increased, add sell market orders
                    if markets[market].asset == "Simula 500":
                        print("PChange selling " + str(quantity*10) + " shares in " + markets[market].asset)
                    self.placeOrder(markets[market], "sell", current_price-0.01, quantity*10, "market")
                else:
                    # Price decreased, add buy market orders
                    if markets[market].asset == "Simula 500":
                        print("PChange buying " + str(quantity*10) + " shares in " + markets[market].asset)
                    self.placeOrder(markets[market], "buy", current_price+0.01, quantity*10, "market")

        # trade off 300 period moving average
        if len(price_list) > 300:
            ma_300 = sum(price_list[-300:]) / 300
            if current_price > ma_300:
                # Price is above 300 period MA, add sell market orders
                if markets[market].asset == "Simula 500":
                    print("300MA selling " + str(quantity) + " shares in " + markets[market].asset)
                self.placeOrder(markets[market], "sell", current_price, quantity, "market")
            else:
                # Price is below 300 period MA, add buy market orders
                if markets[market].asset == "Simula 500":
                    print("300MA buying " + str(quantity) + " shares in " + markets[market].asset)
                self.placeOrder(markets[market], "buy", current_price, quantity, "market")

        # Calculate 8-period moving average
        if len(price_list) >= 8:
            ma_8 = sum(price_list[-8:]) / 8
            
            # Determine action based on current price relative to 8-period MA
            if current_price < ma_8:
                if ma_8 - current_price <  spreads_by_market[market]*5:
                    # Buy market
                    self.placeOrder(markets[market], "buy", current_price, quantity, "market")
                else:
                    # Sell limit
                    self.placeOrder(markets[market], "sell", current_price, quantity, "limit")
            else:
                if current_price - ma_8 < spreads_by_market[market]*5:
                    # Sell market
                    self.placeOrder(markets[market], "sell", current_price, quantity, "market")
                else:
                    # Buy limit
                    self.placeOrder(markets[market], "buy", current_price, quantity, "limit")
        
        if len(price_list) > 30:
            ma_30 = sum(price_list[-30:]) / 30
            if current_price < ma_30:
                # Sell market
                self.placeOrder(markets[market], "sell", current_price, quantity/10, "limit")
            else:
                # Buy market
                self.placeOrder(markets[market], "buy", current_price, quantity/10, "limit")

        # Limit the number of limitorders placed to save on compute
        max_orders = 20
        if len(markets[market].bids) + len(markets[market].asks) > max_orders:
            # Cancel all TA orders
            for bid_level in markets[market].bids.values():
                bid_level.cancelOrdersFromID("TA Trading Firm")
            for ask_level in markets[market].asks.values():
                ask_level.cancelOrdersFromID("TA Trading Firm")
            return

        # place sell limit order at nearest vwap band above, and buy limit order at nearest vwap band below
        upper_band = std_dev * 3 + mean_price
        lower_band = std_dev * -3 + mean_price

        self.placeOrder(markets[market], "sell", upper_band, quantity, "limit")
        self.placeOrder(markets[market], "buy", lower_band, quantity, "limit") 