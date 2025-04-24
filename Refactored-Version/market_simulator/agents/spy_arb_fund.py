from market_simulator.agents.executional_trader import ExecutionalTrader
from market_simulator.utils.market_utils import markets, last_prices, SPREADS
from market_simulator.config import NUM_SHARES, SPY_INCLUDED_ASSETS, ARBITRAGE_THRESHOLD, ARB_QUANTITY

class SpyArbFund(ExecutionalTrader):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)
        self.market_caps = {}
        self.spy_composition = {}
        self.navps = 500
        self._calculate_market_caps()
        self._update_weights()
    
    def _calculate_market_caps(self):
        """Calculate market cap for each constituent stock"""
        for asset in SPY_INCLUDED_ASSETS:
            price = markets[asset].last_price
            shares = NUM_SHARES[asset]
            self.market_caps[asset] = price * shares

    def _update_navps(self):
        """Calculate the NAV per share of the SPY"""
        total = sum(self.market_caps.values())
        self.navps = total / 4000000000

    def _cancel_old(self):
        """Cancel all old orders"""
        for asset in SPY_INCLUDED_ASSETS:
            self.cancelAllOrders(markets[asset])
        
        self.cancelAllOrders(markets["SPY"])

    def _update_weights(self):
        """Calculate the number of shares of each constituent per SPY share"""
        
        for asset in SPY_INCLUDED_ASSETS:
            weight = ( self.market_caps[asset] / 4000000000 ) / markets[asset].last_price
            self.spy_composition[asset] = weight * ARB_QUANTITY

    def update_calculations(self):
        """Update market caps and composition calculations"""
        self._calculate_market_caps()
        self._cancel_old()
        self._update_navps()
        self._update_weights()

    def arbitrage(self):
        """Execute arbitrage if profitable opportunity exists"""
        # Update calculations
        self.update_calculations()
        
        spy_bid = markets["SPY"].get_best_bid()
        spy_ask = markets["SPY"].get_best_ask()

        # If basket is cheaper than SPY by more than arbitrage threshold, buy basket and sell SPY
        if self.navps < spy_bid - ARBITRAGE_THRESHOLD:
            # Sell SPY
            self.placeOrder(
                markets["SPY"],
                "sell",
                0.01,
                ARB_QUANTITY,
                "market"
            )
            
            # Buy constituent stocks
            for asset in SPY_INCLUDED_ASSETS:
                self.placeOrder(
                    markets[asset],
                    "buy",
                    0.01,
                    int(self.spy_composition[asset]),
                    "market"
                )
                
        # If basket is more expensive than SPY by more than arbitrage threshold, sell basket and buy SPY
        elif self.navps > spy_ask + ARBITRAGE_THRESHOLD:
            # Buy SPY
            self.placeOrder(
                markets["SPY"],
                "buy",
                0.01,
                ARB_QUANTITY,
                "market"
            )
            
            # Sell constituent stocks
            for asset in SPY_INCLUDED_ASSETS:
                self.placeOrder(
                    markets[asset],
                    "sell",
                    0.01,
                    int(self.spy_composition[asset]),
                    "market"
                )