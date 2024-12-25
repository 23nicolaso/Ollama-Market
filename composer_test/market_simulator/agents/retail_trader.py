import random
import math
from market_simulator.agents.base_agent import MarketAgent
from market_simulator.config import ASSETS

class RetailTrader(MarketAgent):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)
        self.retailSentimentScore = {asset: 0.5 for asset in ASSETS}
        self.newsUrgency = 1
        # Add new attributes for cyclical behavior with smaller amplitudes
        self.cycle_frequencies = {
            asset: {
                'fast': random.uniform(0.01, 0.02),    # Fast cycle (100-200 ticks)
                'medium': random.uniform(0.002, 0.005), # Medium cycle (500-1000 ticks) 
                'slow': random.uniform(0.0005, 0.001)   # Slow cycle (2000-4000 ticks)
            } for asset in ASSETS
        }
        self.cycle_amplitudes = {
            asset: {
                'fast': random.uniform(0.02, 0.03),    # Smaller amplitude
                'medium': random.uniform(0.03, 0.04),   # Smaller amplitude
                'slow': random.uniform(0.04, 0.05)      # Smaller amplitude
            } for asset in ASSETS
        }
        self.tick_counter = 0

    def get_cyclical_influence(self, asset):
        # Calculate combined wave effect
        self.tick_counter += 1
        wave = (
            math.sin(self.tick_counter * self.cycle_frequencies[asset]['fast']) * self.cycle_amplitudes[asset]['fast'] +
            math.sin(self.tick_counter * self.cycle_frequencies[asset]['medium']) * self.cycle_amplitudes[asset]['medium'] +
            math.sin(self.tick_counter * self.cycle_frequencies[asset]['slow']) * self.cycle_amplitudes[asset]['slow']
        )
        return wave

    def trade(self, orderBook):
        try:
            # Get base sentiment and cyclical influence
            base_sentiment = self.retailSentimentScore[orderBook.asset]
            cyclical_influence = max(-0.01, min(0.01, self.get_cyclical_influence(orderBook.asset))) * 10
            
            # Calculate final sentiment
            final_sentiment = max(0.1, min(0.9, base_sentiment + cyclical_influence))
            
            # Determine trade direction based on final sentiment
            direction = "buy" if final_sentiment > random.random() else "sell"
        except KeyError:
            direction = "buy" if random.random() < 0.5 else "sell"
            final_sentiment = 0.5

        price = orderBook.getLastPrice()+round(random.uniform(-0.05, 0.05), 2)
        sentiment_diff = abs(final_sentiment - 0.5)  # How far sentiment is from neutral
        base_quantity = random.randint(1, 100)
        quantity = int(base_quantity * (1 + sentiment_diff * 4))  # Scale up quantity based on sentiment difference
        position = self.account.getPosition(orderBook.asset)
        order_type = random.choice(["market", "limit"])

        if quantity > 0:
            if direction == "buy":
                if position + quantity < 100000:
                    self.placeOrder(orderBook, direction, price, quantity, order_type)
            else:
                if position - quantity > 0: 
                    self.placeOrder(orderBook, direction, price, quantity, order_type)

    def setReversionUrgency(self, urgency):
        """Sets how quickly sentiment should revert to mean after news events"""
        self.newsUrgency = urgency
        
    def shiftSentimentToMean(self):
        """Gradually shifts both base sentiment and cyclical sentiment towards the mean"""
        for asset in self.retailSentimentScore:
            # Shift base sentiment towards mean
            current_base = self.retailSentimentScore[asset]
            base_shift = (0.5 - current_base) / (500 * self.newsUrgency)
            self.retailSentimentScore[asset] += base_shift
            
            # Ensure base sentiment stays within bounds
            self.retailSentimentScore[asset] = max(0.1, min(0.9, self.retailSentimentScore[asset]))

        # Reduce news urgency if all sentiments are near mean
        if all(0.45 <= self.retailSentimentScore[asset] <= 0.55 for asset in self.retailSentimentScore):
            if self.newsUrgency > 1:
                self.newsUrgency -= 1

    def estimateSentiment(self, orderBook):
        """Estimates current sentiment including cyclical influences"""
        base = self.retailSentimentScore[orderBook.asset]
        cyclical = self.get_cyclical_influence(orderBook.asset)
        variation = random.uniform(-0.05, 0.05)  # Add a small random variation
        
        # Combine base sentiment, cyclical influence, and variation
        sentiment = base + (cyclical * 0.3) + variation  # Reduce cyclical influence
        return max(0.1, min(0.9, sentiment))  # Ensure the result is between 0.1 and 0.9
    
    def estimateImportance(self):
        """Estimates importance of current market conditions"""
        importance = self.newsUrgency
        variation = random.randint(-2, 2)  # Add a small random variation
        return max(0, min(importance + variation, 10)) 