from market_simulator.agents.executional_trader import ExecutionalTrader
from market_simulator.utils.market_utils import markets, price_history
from market_simulator.config import SPREADS, HFT_BASE_ORDER_SIZE, HFT_POSITION_LIMIT

class HFTFund(ExecutionalTrader):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)
        self.intendedOrders = {}

    def tradeMicrostructure(self, ticker):
        # Check for order book imbalance
        order_book = markets[ticker]
        total_bid_size = order_book._bidSize
        total_ask_size = order_book._askSize
        
        # Calculate imbalance ratio
        if total_bid_size > 0 and total_ask_size > 0:
            imbalance_ratio = total_bid_size / total_ask_size
            
            # If significant imbalance (more than 2x), front run with market order
            if imbalance_ratio > 2:
                # Front run with buy order when there's excess demand
                current_position = self.account.getPosition(ticker)
                max_buy = HFT_POSITION_LIMIT - current_position
                if max_buy > 0:
                    quantity = min(HFT_BASE_ORDER_SIZE * 2, total_bid_size // 4, max_buy)
                    if int(quantity) > 0:
                        self.placeOrder(order_book, "buy", 0.01, int(quantity), "market")
                
            elif imbalance_ratio < 0.5:
                # Front run with sell order when there's excess supply
                current_position = self.account.getPosition(ticker)
                max_sell = HFT_POSITION_LIMIT + current_position
                if max_sell > 0:
                    quantity = min(HFT_BASE_ORDER_SIZE * 2, total_ask_size // 4, max_sell)
                    if int(quantity) > 0:
                        self.placeOrder(order_book, "sell", 0.01, int(quantity), "market")

    def tradeTheNews(self, market, sentiment_score):
        #  If sentiment is above 0.7 or below 0.3, make the HFT front run the trade by market buying/selling 
        if sentiment_score <= 0.3:
            current_price = markets[market].lastPrice
            quantity = HFT_BASE_ORDER_SIZE * (0.3-max(0,sentiment_score))*10

            self.placeOrder(markets[market], "sell", current_price, int(quantity), "market")

        elif sentiment_score >= 0.7:
            current_price = markets[market].lastPrice
            quantity = HFT_BASE_ORDER_SIZE * (1-min(1,sentiment_score))*10

            self.placeOrder(markets[market], "buy", current_price, int(quantity), "market")