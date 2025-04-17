# account.pyx
cdef class Account:
    cdef public int accountID
    cdef dict positions
    cdef double initial_cash

    def __init__(self, int accountID, double CASH):
        self.accountID = accountID
        self.positions = {"CASH": CASH}
        self.initial_cash = CASH

    cpdef void addPosition(self, str asset, double quantity):
        self.positions[asset] = self.positions.get(asset, 0.0) + quantity

    cpdef double getPosition(self, str asset):
        return self.positions.get(asset, 0.0)

    cpdef double getCash(self):
        return self.positions["CASH"]

    cpdef double getValue(self):
        from market_simulator.utils.market_utils import last_prices
        cdef double total = self.positions["CASH"]
        for asset in self.positions:
            if asset != "CASH":
                total += self.positions[asset] * last_prices[asset]
        return total

    cpdef void tradeAtPrice(self, str asset, double price, double quantity, int direction):
        self.addPosition(asset, quantity * direction)
        self.addPosition("CASH", -price * quantity * direction)
