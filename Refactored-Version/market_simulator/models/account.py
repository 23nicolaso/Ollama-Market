class Account:
    def __init__(self, accountID, CASH):
        self.accountID = accountID
        self.positions = {"CASH": CASH}
        self.initial_cash = CASH

    def addPosition(self, asset, quantity):
        if asset in self.positions:
            self.positions[asset] += quantity
        else:
            self.positions[asset] = quantity

    def getPosition(self, asset):
        return self.positions.get(asset, 0)

    def getCash(self):
        return self.positions["CASH"]

    def getValue(self):
        from market_simulator.utils.market_utils import last_prices
        return self.positions["CASH"] + sum([self.positions[asset] * last_prices[asset] for asset in self.positions if asset != "CASH"])

    def tradeAtPrice(self, asset, price, quantity, direction):
        self.addPosition(asset, quantity*direction)
        self.addPosition("CASH", -price*quantity*direction) 