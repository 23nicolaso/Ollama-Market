# Market's current state is stored as a HMM
"""
States:
Inflation state (strong dec/dec/neutral/inc/strong inc)
interest rate (-0.5, -0.25, 0, 0.25, 0.5, probabilities derived from inflation state, mostly flat)
stock risk (low, medium, high)
overall economic strength (low, medium, high, derived from combination of jobs state, inflation, interest rates) 
sector by sector expected profits (some correlation between sectors, some based on inflation, sectors should have generic return profiles based on economic strength, along with individual company risks). 
Should also have small number of emissions, which aren't treated as news which is traded, and is very noisy data, but is released on regular basis and indicates state.
"""

class MarketState:
    def __init__(self):
        self.expectedInflationRate = 0.03
        self.federalInterestRate = 0.05
        self.stockRiskPremium = 0.03
        self.economicStrength = None
        self.unemploymentRate = 0.042
        self.head = None # allocate one node to each stock, with risk/reward, group nodes by sector, group sector into general category
        # basically a huge tree of markov states
    def updateStateRandomly(self):
        return
    def updateStateWithNews(self):
        return
    def getValue(self, ticker):
        return