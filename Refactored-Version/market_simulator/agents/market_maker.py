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
        volatility_factor = min(10.0, max(1.0, (std_dev / mean_price) * 1000))  # Cap at 5x spread widening

        # Adjust base spread for volatility
        baseSpread = baseSpread * volatility_factor

        # Calculate position skew
        position = self.account.getPosition(orderBook.asset)
        position_limit = MM_POSITION_LIMIT
        skew = min(max(-0.5, position / position_limit), 0.5)  # Skew ranges from -0.5 to 0.5
        
        # Adjust prices based on position skew
        bidPrice = round(midPrice - (baseSpread/2 * (1 + skew)), 2)
        askPrice = round(midPrice + (baseSpread/2 * (1 - skew)), 2)
        bq, aq = orderBook.getNearbyDepth(NEARBY_RANGE)
        # Layer orders at different sizes and prices
        # if order book is illiquid not matching liquidity requirements in one direction, quote orders
        for i in range(MM_DEPTH):
            # Reduce size during high volatility
            layerSize = int(MM_BASE_ORDER_SIZE * (i + 1) / volatility_factor)
            
            if bq < MM_REQUIRED_LIQ:
                bidLayerPrice = round(bidPrice - (0.01 * i), 2)
                self.placeOrder(orderBook, "buy", bidLayerPrice, layerSize, "limit")
            if aq < MM_REQUIRED_LIQ:
                askLayerPrice = round(askPrice + (0.01 * i), 2)
                self.placeOrder(orderBook, "sell", askLayerPrice, layerSize, "limit")

    def provideLiquidity(self, orderBook):
        remaining_urgent_buys, remaining_urgent_sells = orderBook.getUrgentQuantity()

        if remaining_urgent_buys > 0:
            # print("SELLING liquidity to the market")
            price_change = self.spreads[orderBook.asset] * remaining_urgent_buys / (MM_DEPTH*MM_BASE_ORDER_SIZE) # Adjust the divisor as needed
            self.placeOrder(orderBook, "sell", orderBook.lastPrice + round(price_change, 2), remaining_urgent_buys, "limit")

        if remaining_urgent_sells > 0:
            # print("BUYING liquidity from the market")
            price_change = self.spreads[orderBook.asset] * remaining_urgent_sells / (MM_DEPTH*MM_BASE_ORDER_SIZE) # Adjust the divisor as needed
            self.placeOrder(orderBook, "buy", orderBook.lastPrice - round(price_change, 2), remaining_urgent_sells, "limit")