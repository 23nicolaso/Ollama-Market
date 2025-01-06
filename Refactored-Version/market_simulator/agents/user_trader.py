from market_simulator.agents.executional_trader import ExecutionalTrader
from market_simulator.utils.market_utils import markets

class UserTrader(ExecutionalTrader):
    def __init__(self, account_id="USER_001", cash=10000000):
        super().__init__(account_id, cash)

    def execute_order(self, order):
        if order.order_type == "iceberg":
            self.executeTradeInLegs(markets[order.market], order.direction, order.price, order.quantity)
        else:
            self.placeOrder(markets[order.market], order.direction, order.price, order.quantity, order.order_type)
