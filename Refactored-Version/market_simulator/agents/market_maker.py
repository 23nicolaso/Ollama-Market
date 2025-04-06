from market_simulator.agents.base_agent import MarketAgent
from market_simulator.utils.market_utils import price_history
from market_simulator.config import MM_POSITION_LIMIT, MM_BASE_ORDER_SIZE, MM_DEPTH, NEARBY_RANGE, MM_REQUIRED_LIQ

class MarketMaker(MarketAgent):
    def __init__(self, accountID, cash, spreads):
        super().__init__(accountID, cash)
        self.spreads = spreads

    def wipeAllOrders(self, orderBook):
        self.cancelAllOrders(orderBook)

    def makeMarket(self, orderBook):
        # Get current market state
        midPrice = orderBook.lastPrice
        self.wipeAllOrders(orderBook)

        # Calculate base spread
        baseSpread = self.spreads[orderBook.asset]

        # Calculate volatility based on recent price history
        mean_price = price_history[orderBook.asset].mean()
        std_dev = price_history[orderBook.asset].std()
        volatility_factor = min(10.0, max(1.0, (std_dev / mean_price) * 1000))

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
            # Reduce size during high volatility
            layerSize = int(MM_BASE_ORDER_SIZE * (i + 1) / volatility_factor)
    
            bidLayerPrice = round(bidPrice - (0.01 * i), 2)
            self.placeOrder(orderBook, "buy", bidLayerPrice, layerSize, "limit")
        
            askLayerPrice = round(askPrice + (0.01 * i), 2)
            self.placeOrder(orderBook, "sell", askLayerPrice, layerSize, "limit")

    def provideLiquidity(self, orderBook):
        remaining_urgent_buys, remaining_urgent_sells = orderBook.getUrgentQuantity()

        if remaining_urgent_buys > 0:
            price_change = self.spreads[orderBook.asset] * remaining_urgent_buys / (MM_DEPTH*MM_BASE_ORDER_SIZE)
            self.placeOrder(orderBook, "sell", orderBook.lastPrice + round(price_change, 2), remaining_urgent_buys, "limit")

        if remaining_urgent_sells > 0:
            price_change = self.spreads[orderBook.asset] * remaining_urgent_sells / (MM_DEPTH*MM_BASE_ORDER_SIZE)
            self.placeOrder(orderBook, "buy", orderBook.lastPrice - round(price_change, 2), remaining_urgent_sells, "limit")