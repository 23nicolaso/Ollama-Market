from market_simulator.agents.executional_trader import ExecutionalTrader
from market_simulator.utils.market_utils import price_history, markets, assets, action_queue, last_prices
from market_simulator.config import RISK_ON_ASSETS, RISK_OFF_ASSETS, INITIAL_PRICES

# Dollar budget per asset when rotating (scaled to ~80% of initial $10M / 5 assets)
RISK_PER_ASSET_DOLLARS = 1_600_000

class RiskOnRiskOffFirm(ExecutionalTrader):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)
        # Start risk-on: buy each risk-on asset with dollar budget, cash deducted
        for asset in RISK_ON_ASSETS:
            shares = int(RISK_PER_ASSET_DOLLARS / INITIAL_PRICES[asset])
            if shares > 0:
                cost = shares * INITIAL_PRICES[asset]
                self.account.addPosition(asset, shares)
                self.account.addPosition("CASH", -cost)

    def _target_shares(self, asset):
        """Dollar-budget → share count at current price."""
        price = max(last_prices.get(asset, INITIAL_PRICES[asset]), 0.01)
        return int(RISK_PER_ASSET_DOLLARS / price)

    def risk_off(self):
        action_queue.put("Risk Mgr: RISK OFF — rotating into HEALTHCARE / ENERGY / GOLD / TBILLS")
        for asset in RISK_ON_ASSETS:
            self.targetPosition(markets[asset], "sell", markets[asset].last_price-0.1, markets[asset].last_price, 0, True)
        for asset in RISK_OFF_ASSETS:
            self.targetPosition(markets[asset], "buy", markets[asset].last_price+0.1, markets[asset].last_price, self._target_shares(asset), True)

    def risk_on(self):
        action_queue.put("Risk Mgr: RISK ON — rotating into TECH / CONSUMER / FINANCIAL / BITCOIN")
        for asset in RISK_ON_ASSETS:
            self.targetPosition(markets[asset], "buy", markets[asset].last_price+0.1, markets[asset].last_price, self._target_shares(asset), True)
        for asset in RISK_OFF_ASSETS:
            self.targetPosition(markets[asset], "sell", markets[asset].last_price-0.1, markets[asset].last_price, 0, True)
