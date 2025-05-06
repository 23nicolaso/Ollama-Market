from market_simulator.agents.spy_arb_fund import SpyArbFund
from market_simulator.utils.market_utils import price_history
from market_simulator.config import MM_POSITION_LIMIT, MM_BASE_ORDER_SIZE, MM_DEPTH, NEARBY_RANGE

class MarketMaker(SpyArbFund):
    def __init__(self, accountID, cash, spreads):
        super().__init__(accountID, cash)
        self.spreads = spreads

    def wipeAllOrders(self, orderBook):
        self.cancelAllOrders(orderBook)

    def makeMarket(self, orderBook):
        # Get current market state
        self.wipeAllOrders(orderBook)

        # Calculate base spread
        baseSpread = self.spreads[orderBook.asset]

        if orderBook.asset == "SPY":
            self.update_calculations()
            fair_price = self.navps
            midPrice = fair_price
        else:
            if len(price_history) > 50:
                midPrice = price_history[orderBook.asset].mean(n=50) # use mean of last 50 trades for fair price estimate
            else:
                midPrice = orderBook.last_price

        # Calculate volatility based on recent price history
        mean = price_history[orderBook.asset].mean()
        std = price_history[orderBook.asset].std()
        
        volatility_factor = min(5.0, max(1.0, (std / mean) * 1000))

        # Adjust base spread for volatility
        baseSpread = baseSpread * volatility_factor

        # Calculate inventory skew
        current_position = self.account.getPosition(orderBook.asset)
        position_threshold = 0.25 * MM_POSITION_LIMIT
        skew_factor = 0
        
        if current_position > position_threshold:
            # Long inventory - skew down
            skew_factor = min(1.0, (current_position - position_threshold) / MM_POSITION_LIMIT)
            midPrice = midPrice * (1 - skew_factor * 0.01)  # Reduce mid price by up to 1%
        elif current_position < -position_threshold:
            # Short inventory - skew up
            skew_factor = min(1.0, abs(current_position + position_threshold) / MM_POSITION_LIMIT)
            midPrice = midPrice * (1 + skew_factor * 0.01)  # Increase mid price by up to 1%

        bidPrice = round(midPrice - (baseSpread), 2)
        askPrice = round(midPrice + (baseSpread), 2)

        for i in range(MM_DEPTH):
            layerSize = int(MM_BASE_ORDER_SIZE * (i + 1))
    
            bidLayerPrice = round(bidPrice - (0.01 * i), 2)
            self.placeOrder(orderBook, "buy", bidLayerPrice, layerSize, "limit")
        
            askLayerPrice = round(askPrice + (0.01 * i), 2)
            self.placeOrder(orderBook, "sell", askLayerPrice, layerSize, "limit")

    def provideLiquidity(self, orderBook):
        remaining_urgent_buys = orderBook.get_market_buy_quantity()
        remaining_urgent_sells = orderBook.get_market_sell_quantity()

        if remaining_urgent_buys > 0:
            price_change = self.spreads[orderBook.asset] * remaining_urgent_buys / (MM_DEPTH*MM_BASE_ORDER_SIZE*10)
            self.placeOrder(orderBook, "sell", orderBook.last_price + round(price_change, 2), remaining_urgent_buys, "limit")

        if remaining_urgent_sells > 0:
            price_change = self.spreads[orderBook.asset] * remaining_urgent_sells / (MM_DEPTH*MM_BASE_ORDER_SIZE*10)
            self.placeOrder(orderBook, "buy", orderBook.last_price - round(price_change, 2), remaining_urgent_sells, "limit")