from market_simulator.agents.base_agent import MarketAgent
from market_simulator.utils.market_utils import estimateUnderlyingValue, price_history
from market_simulator.config import MM_POSITION_LIMIT, MM_BASE_ORDER_SIZE, MM_DEPTH

class MarketMaker(MarketAgent):
    def __init__(self, accountID, cash, spreads):
        super().__init__(accountID, cash)
        self.spreads = spreads

    def wipeOldOrders(self, orderBook):
        value = estimateUnderlyingValue(orderBook.asset)

        for bidLevel in orderBook.bids.values():
            if value - bidLevel.getPrice() >= self.spreads[orderBook.asset] * MM_DEPTH:
                bidLevel.cancelOrdersFromID("MARKET MAKER")
        
        for askLevel in orderBook.asks.values():
            if askLevel.getPrice() - value >= self.spreads[orderBook.asset] * MM_DEPTH:
                askLevel.cancelOrdersFromID("MARKET MAKER")

    def wipeAllOrders(self, orderBook):
        for bidLevel in orderBook.bids.values():
            bidLevel.cancelOrdersFromID("MARKET MAKER")
        for askLevel in orderBook.asks.values():
            askLevel.cancelOrdersFromID("MARKET MAKER")

    def makeMarket(self, orderBook):
        # Get current market state

        midPrice = orderBook.getLastPrice()

        # Clear existing orders
        self.wipeAllOrders(orderBook)

        # Calculate base spread
        baseSpread = self.spreads[orderBook.asset]

        # Calculate volatility based on recent price history
        price_list = price_history[orderBook.asset][-50:]
        mean_price = sum(price_list) / len(price_list)
        std_dev = (sum((x - mean_price) ** 2 for x in price_list) / len(price_list)) ** 0.5
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

        # Layer orders at different sizes and prices
        for i in range(MM_DEPTH):
            # Reduce size during high volatility
            layerSize = int(MM_BASE_ORDER_SIZE * (i + 1) / volatility_factor)
            
            bidLayerPrice = round(bidPrice - (0.01 * i), 2)
            askLayerPrice = round(askPrice + (0.01 * i), 2)
            
            # Place orders with size adjusted by distance from mid
            self.placeOrder(orderBook, "buy", bidLayerPrice, layerSize, "limit")
            self.placeOrder(orderBook, "sell", askLayerPrice, layerSize, "limit")

    def provideLiquidity(self, orderBook):
        remaining_urgent_buys, remaining_urgent_sells = orderBook.getUnfilledUrgentOrders()
        total_buy_quantity = sum(order[0] for order in remaining_urgent_buys if order[1] != "MARKET MAKER")
        total_sell_quantity = sum(order[0] for order in remaining_urgent_sells if order[1] != "MARKET MAKER")

        if total_buy_quantity > 0:
            price_change = self.spreads[orderBook.asset] * total_buy_quantity / (MM_DEPTH*MM_BASE_ORDER_SIZE) # Adjust the divisor as needed
            self.placeOrder(orderBook, "sell", orderBook.getLastPrice() + round(price_change, 2), total_buy_quantity, "limit")

        if total_sell_quantity > 0:
            price_change = self.spreads[orderBook.asset] * total_sell_quantity / (MM_DEPTH*MM_BASE_ORDER_SIZE) # Adjust the divisor as needed
            self.placeOrder(orderBook, "buy", orderBook.getLastPrice() - round(price_change, 2), total_sell_quantity, "limit")

        orderBook.fillUrgentOrders() 