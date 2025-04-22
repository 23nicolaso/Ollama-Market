import random
import math
from market_simulator.agents.base_agent import MarketAgent
from market_simulator.config import ASSETS, RETAIL_POSITION_LIMIT, RETAIL_MAX_ORDER_SIZE, USE_CYCLICAL_SENTIMENT, SENTIMENT_REVERSION_RATE
from market_simulator.utils.market_utils import calculate_fair_value

class RetailTrader(MarketAgent):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)
        self.retailSentimentScore = {asset: 0.5 for asset in ASSETS}
        self.newsUrgency = 1
        self.tick_counter = 0

    def trade(self, orderBook):
        try:
            # Get base sentiment and cyclical influence
            base_sentiment = self.retailSentimentScore[orderBook.asset]
            
            # Calculate final sentiment
            final_sentiment = max(0.25, min(0.75, base_sentiment))
            
            # Determine trade direction based on final sentiment
            direction = "buy" if final_sentiment > random.random() else "sell"
        except KeyError:
            direction = "buy" if random.random() < 0.5 else "sell"
            final_sentiment = 0.5
    
        bid = orderBook.get_best_bid()
        ask = orderBook.get_best_ask()

        sentiment_diff = abs(final_sentiment - 0.5)  # How far sentiment is from neutral
        base_quantity = random.randint(1, RETAIL_MAX_ORDER_SIZE)
        s_quantity = int(base_quantity * (1 + sentiment_diff * 4))  # Scale up quantity based on sentiment difference
        position = self.account.getPosition(orderBook.asset)
        type = random.choice(["market","limit"])
        quantity = s_quantity if type == "market" else s_quantity // 2

        if quantity > 0:
            if direction == "buy":
                if position + quantity < RETAIL_POSITION_LIMIT:
                    self.placeOrder(orderBook, "buy", bid+random.choice([-0.02,-0.01,0,0.01,0.02]), quantity, type)
            else:
                if position - quantity > 0: 
                    self.placeOrder(orderBook, "sell", ask-random.choice([-0.02,-0.01,0,0.01,0.02]), quantity, type)

    def setReversionUrgency(self, urgency):
        """Sets how quickly sentiment should revert to mean after news events"""
        self.newsUrgency = max(1, urgency)
        
    def shiftSentimentToMean(self):
        """Gradually shifts both base sentiment and cyclical sentiment towards the mean"""
        for asset in self.retailSentimentScore:
            # Shift base sentiment towards mean
            current_base = self.retailSentimentScore[asset]
            base_shift = (0.5 - current_base) / (SENTIMENT_REVERSION_RATE * self.newsUrgency)
            self.retailSentimentScore[asset] += base_shift
            
            # Ensure base sentiment stays within bounds
            self.retailSentimentScore[asset] = max(0.1, min(0.9, self.retailSentimentScore[asset]))

            if self.retailSentimentScore[asset] > 0.49 and self.retailSentimentScore[asset] < 0.51:
                self.retailSentimentScore[asset] += random.uniform(-0.1, 0.1)

        # Reduce news urgency if all sentiments are near mean
        if all(0.45 <= self.retailSentimentScore[asset] <= 0.55 for asset in self.retailSentimentScore):
            if self.newsUrgency > 1:
                self.newsUrgency = 1

    def estimateSentiment(self, orderBook):
        """Estimates current sentiment including cyclical influences"""
        base = self.retailSentimentScore[orderBook.asset]
        if USE_CYCLICAL_SENTIMENT:
            cyclical = self.get_cyclical_influence(orderBook.asset)
        else:
            cyclical = 0
        variation = random.uniform(-0.05, 0.05)  # Add a small random variation
        
        # Combine base sentiment, cyclical influence, and variation
        sentiment = base + (cyclical * 0.3) + variation  # Reduce cyclical influence
        return max(0.1, min(0.9, sentiment))  # Ensure the result is between 0.1 and 0.9
    
    def estimateImportance(self):
        """Estimates importance of current market conditions"""
        importance = self.newsUrgency
        variation = random.randint(-2, 2)  # Add a small random variation
        return max(0, min(importance + variation, 10)) 