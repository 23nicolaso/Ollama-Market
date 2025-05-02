from market_simulator.agents.executional_trader import ExecutionalTrader
from market_simulator.utils.market_utils import price_history, markets, assets
from market_simulator.config import RISK_ON_ASSETS, RISK_OFF_ASSETS, RISK_FIRM_POSITION_SIZE
class RiskOnRiskOffFirm(ExecutionalTrader):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)
        for asset in RISK_ON_ASSETS:
            self.account.addPosition(asset, RISK_FIRM_POSITION_SIZE) # Start with risk on mode

    def risk_off(self):
        for asset in RISK_ON_ASSETS:
            self.targetPosition(markets[asset], "sell", markets[asset].last_price-0.1, markets[asset].last_price, 0, True)
        for asset in RISK_OFF_ASSETS:
            self.targetPosition(markets[asset], "buy", markets[asset].last_price+0.1, markets[asset].last_price, RISK_FIRM_POSITION_SIZE, True)

    def risk_on(self):
        for asset in RISK_ON_ASSETS:
            self.targetPosition(markets[asset], "buy", markets[asset].last_price+0.1, markets[asset].last_price, RISK_FIRM_POSITION_SIZE, True)
        for asset in RISK_OFF_ASSETS:
            self.targetPosition(markets[asset], "sell", markets[asset].last_price-0.1, markets[asset].last_price, 0, True)
